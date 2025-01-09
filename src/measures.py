import scipy
import torch
import numpy as np

from collections import defaultdict
from typing import Optional, List

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

class Scorer:

    def __init__(self, model_name_or_path: str, device: Optional[str] = "cuda"):
        """
        :param model_name_or_path (str):
            the name or path to a model compatible with AutoModelWithLMHead
        :param device (str):
            "cpu", "cuda", or "mps"
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")

        # get the model's context window size
        if "n_positions" in self.model.config.__dict__.keys():
            self.max_seq_len = self.model.config.n_positions
        else:
            self.max_seq_len = self.model.config.max_position_embeddings

    @staticmethod
    def get_word_character_positions(string):
        """
        Takes a string and returns a list of tuples, each containing the start and end character positions of a word in the string.
        :param string (str):
            the input string
        """
        words = string.split()
        char_index_pairs = []
        pos = 0
        for w in words:
            char_index_pairs.append((pos, pos + len(w)))
            pos += len(w) + 1
        return char_index_pairs

    def aggregate_score_by_word(self, string, scores, offsets, mode):
        try:
            scores = [s.item() for s in scores]
        except AttributeError:
            pass

        agg_scores = []
        word_mapping = self.get_word_character_positions(string)
        current_word_index = 0
        if mode == 'first':
            for score, index in zip(scores, offsets):
                start, end = index
                start_current_word, end_current_word = word_mapping[current_word_index]
                if (start + 1 == start_current_word) or (start == start_current_word == 0):
                    agg_scores.append(score)
                if end == end_current_word:
                    current_word_index += 1
                # assert end <= end_current_word
        elif mode == 'sum':
            current_score = 0
            for score, index in zip(scores, offsets):
                current_score += score
                start, end = index
                start_current_word, end_current_word = word_mapping[current_word_index]
                if end == end_current_word:
                    agg_scores.append(current_score)
                    current_score = 0
                    current_word_index += 1
                # assert end <= end_current_word
        elif mode == 'mean':
            current_score = []
            for score, index in zip(scores, offsets):
                current_score.append(score)
                start, end = index
                start_current_word, end_current_word = word_mapping[current_word_index]
                if end == end_current_word:
                    agg_scores.append(np.mean(current_score))
                    current_score = []
                    current_word_index += 1
        elif mode == 'string':
            current_score = []
            for score, index in zip(scores, offsets):
                current_score.append(score)
                start, end = index
                start_current_word, end_current_word = word_mapping[current_word_index]
                if end == end_current_word:
                    agg_scores.append(self.tokenizer.decode(current_score).strip())
                    current_score = []
                    current_word_index += 1
        return np.array(agg_scores)


class SurprisalScorer(Scorer):

    def __init__(self, model_name_or_path: str, device: Optional[str] = "cpu"):
        """
        :param model_name_or_path: the name or path to a model
            compatible with AutoModelWithLMHead
        :param device: "cpu" or "cuda"
        """
        super().__init__(model_name_or_path, device)

    def score(
            self,
            input: str,
            aggregate_by_word: Optional[bool] = False,
            return_tokens: Optional[bool] = False
    ):
        """
        :param input (str):
            the input string to obtain surprisal scores for
        :param aggregate_by_word (bool):
            whether to aggregate token surprisal for words spanning multiple tokens
        :param return_tokens (bool):
            whether to return the tokens (or words) corresponding to each surprisal score

        :return (dict):
            dictionary containing token/word level 'surprisal', 'entropy', 'deviation' stored as numpy
            arrays, and 'tokens' (a list of unit_sized strings) if return_tokens.
        """

        # tokenize input string
        input_tokenized = self.tokenizer(input, return_offsets_mapping=aggregate_by_word, return_attention_mask=False)

        input_tokenized['input_ids'] = [self.tokenizer.bos_token_id] + input_tokenized['input_ids']

        # if input sequences longer than context window, drop the last tokens and warn
        if len(input_tokenized["input_ids"]) > self.max_seq_len:
            raise NotImplementedError(
                "Input sequence longer than model's context window. Truncation not yet implemented.")

        # to tensor
        input_ids = torch.tensor(input_tokenized["input_ids"], device=self.device)

        # forward pass
        with torch.no_grad():
            outputs = self.model(input_ids.unsqueeze(0))

        # tranform logits into log probabilities (for the entire vocabulary)
        log_probs = torch.nn.functional.log_softmax(outputs.logits.squeeze(0), dim=-1)

        # next-token log probabilities
        next_token_log_probs = log_probs[
            range(len(input_ids) - 1),
            input_ids[1:],
        ]

        # next-token entropy
        probs = torch.exp(log_probs)
        next_token_entropies = - (log_probs * probs).nansum(-1)
        next_token_entropies = next_token_entropies[range(len(input_ids) - 1)]

        # next-token deviation of surprisal from entropy
        next_token_deviations = (
            - next_token_log_probs
            - next_token_entropies
        )

        # aggregate token-level scores by word
        if aggregate_by_word:
            offsets = input_tokenized['offset_mapping']
            next_token_log_probs = self.aggregate_score_by_word(input, next_token_log_probs, offsets, mode='sum')
            next_token_entropies = self.aggregate_score_by_word(input, next_token_entropies, offsets, mode='first')
            next_token_deviations = self.aggregate_score_by_word(input, next_token_deviations, offsets, mode='first')
            if return_tokens:
                next_tokens = self.aggregate_score_by_word(input, input_tokenized["input_ids"][1:], offsets, mode='string')
        elif return_tokens:
            next_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[1:])

        # construct output dictionary
        rdict = {
            "surprisal": - next_token_log_probs,
            "entropy": next_token_entropies,
            "deviation": next_token_deviations,
        }
        for k in rdict:
            if type(rdict[k]) == torch.Tensor:
                rdict[k] = rdict[k].numpy()
        if return_tokens:
            rdict['tokens'] = next_tokens

        return rdict


class IncrementalInformationValueScorer(Scorer):

    def __init__(
            self,
            model_name_or_path: str,
            device: Optional[str] = "cpu",
            layers: Optional[list] = None,
            temporally_align: Optional[bool] = True,
            summary_fn: Optional[str] = None,
            distance_metric: Optional[str] = "euclidean",
            mean_std_embeddings_path: Optional[str] = None,
            seed: Optional[int] = 0
    ):
        """
        :param model_name_or_path: the name or path to a model
            compatible with AutoModelWithLMHead
        :param device: "cpu" or "cuda"
        """
        super().__init__(model_name_or_path, device)
        self.layers = layers if layers else list(range(self.model.config.n_layer + 1))
        self.temporally_align = temporally_align
        self.summary_fn = summary_fn
        self.distance_metric = distance_metric
        self.standardize_embeddings = mean_std_embeddings_path is not None
        self.seed = seed

        if self.standardize_embeddings:
            self.mean_std_embeds = torch.load(mean_std_embeddings_path)
    
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def aggregate_embeddings(self, hidden_states_batch: torch.Tensor, forecast_horizon: int, seq_len=None,
                             length_true_seq=None):
        """
        :param hidden_states_batch: tensor of shape [n_alternatives, n_tokens, n_dims]
        :param forecast_horizon: the number of tokens to consider
        :return: tensor of shape [n_alternatives, n_dims]
        """
        if forecast_horizon is None:
            raise ValueError("forecast_horizons must be a positive integer")

        if length_true_seq is not None:
            assert seq_len is not None
            forecast_horizon_in_tokens = length_true_seq
            forecast_horizon_in_tokens += (forecast_horizon - 1) * seq_len
        else:
            if seq_len is not None:
                forecast_horizon_in_tokens = seq_len * forecast_horizon
            else:
                forecast_horizon_in_tokens = forecast_horizon

        return hidden_states_batch[:, :forecast_horizon_in_tokens, :].mean(1)

    def embed_alternatives(self, A: list, B: list, forecast_horizons: List[int]):
        if forecast_horizons is None:
            raise ValueError("forecast_horizons must be a list of positive integers")

        flat_list = []
        for i, a in enumerate(A):
            flat_list.append(a[1:] if self.temporally_align else a)
        for i, b in enumerate(B):
            flat_list.append(b[:-1] if self.temporally_align else b)

        # Forward pass with all alternatives
        with torch.no_grad():
            outputs = self.model(torch.stack(flat_list).to(self.device), output_hidden_states=True)
            hidden_states = outputs.hidden_states
            # hidden_states = [hs.detach().cpu() for hs in outputs.hidden_states]

        # Collect alternative embeddings for each layer and forecast horizon
        embeds_A = dict()
        embeds_B = dict()

        for l in self.layers:
            for h in forecast_horizons:
                embeds_A[(l, h)] = self.aggregate_embeddings(hidden_states[l][:len(A)], h)
                embeds_B[(l, h)] = self.aggregate_embeddings(hidden_states[l][len(A):], h)

        return embeds_A, embeds_B

    def embed_seq_alternatives(self, A: list, B: list, forecast_horizons: List[int], seq_len, length_true_seq_A,
                               length_true_seq_B):

        if forecast_horizons is None:
            raise ValueError("forecast_horizons must be a list of positive integers")

        truncated_alternatives_A = []
        for i, a in enumerate(A):
            truncated_alternatives_A.append(a[length_true_seq_A:] if self.temporally_align else a)

        truncated_alternatives_B = []
        for i, b in enumerate(B):
            truncated_alternatives_B.append(b[:-seq_len] if self.temporally_align else b)

        else:
            # Forward pass with all alternatives
            with torch.no_grad():
                outputs_A = self.model(torch.stack(truncated_alternatives_A).to(self.device), output_hidden_states=True)
                hidden_states_A = outputs_A.hidden_states
                outputs_B = self.model(torch.stack(truncated_alternatives_B).to(self.device), output_hidden_states=True)
                hidden_states_B = outputs_B.hidden_states
                # hidden_states = [hs.detach().cpu() for hs in outputs.hidden_states]

            # Collect alternative embeddings for each layer and forecast horizon
            embeds_A = dict()
            embeds_B = dict()

            for l in self.layers:
                for h in forecast_horizons:
                    embeds_A[(l, h)] = self.aggregate_embeddings(hidden_states_A[l], h, seq_len)
                    embeds_B[(l, h)] = self.aggregate_embeddings(hidden_states_B[l], h, seq_len, length_true_seq_B)

            return embeds_A, embeds_B
    
    @staticmethod
    def spearmanr_dist(A, B):
        distances = torch.zeros(A.shape[0], B.shape[0])
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                distances[i, j] = 1 - scipy.stats.spearmanr(A[i], B[j]).statistic
        return distances

    def pairwise_distances(self, A: torch.Tensor, B: torch.Tensor):
        """
        :param A: tensor of shape [n_alternatives_A, n_dims]
        :param B: tensor of shape [n_alternatives_B, n_dims]
        :return: dictionary mapping summary statistics to distance values
        """
        if self.distance_metric not in ["euclidean", "seuclidean", "cosine", "spearmanr", "canberra"]:
            raise ValueError("metric must be one of 'euclidean', 'seuclidean', 'cosine', 'spearmanr', 'canberra'")
        
        if isinstance(A, torch.Tensor):
            A = A.cpu()
        if isinstance(B, torch.Tensor):
            B = B.cpu()

        if self.distance_metric == "spearmanr":
            distance_matrix = self.spearmanr_dist(A, B)
        else:
            distance_matrix = scipy.spatial.distance.cdist(A, B, metric=self.distance_metric)

        if self.summary_fn is None:
            distance_dict = {
                "mean": distance_matrix.mean(),
                "max": distance_matrix.max(),
                "min": distance_matrix.min()
            }
        elif self.summary_fn == "mean":
            distance_dict = {
                "mean": distance_matrix.mean()
            }
        elif self.summary_fn == "max":
            distance_dict = {
                "max": distance_matrix.max()
            }
        elif self.summary_fn == "min":
            distance_dict = {
                "min": distance_matrix.min()
            }
        else:
            raise ValueError("summary_fn must be one of 'mean', 'max', 'min', or None")
        
        return distance_dict
    
    def map_words_to_token_positions(self, string, unit=1, tokenizer_offset_mapping=None, skip_first_token=False):
        """
        Map words or ngrams in a string to token positions in the tokenized string.

        :param string (str):
            the input string
        :param tokenizer (transformers.PreTrainedTokenizer):
            the tokenizer used to tokenize the string
        :param unit (int or str):
            the unit to map to token positions, a positive integer
        :param tokenizer_offset_mapping (list):
            the offset mapping of the tokenized string
        :return (list):
            a list of token positions
        """

        # check if unit is valid
        if type(unit) is not int or unit < 1:
            raise ValueError("unit must be a positive integer")

        # tokenize string and get offset mapping
        if tokenizer_offset_mapping is None:
            tokenizer_offset_mapping = self.tokenizer(
                string, return_offsets_mapping=True, return_attention_mask=False
            )['offset_mapping']

        # get character positions of words in string
        word_character_positions = self.get_word_character_positions(string)
        if skip_first_token:
            word_character_positions = word_character_positions[1:]

        token_positions = []
        for (word_start, word_end) in word_character_positions:
            word_pos = []
            for i, (token_start, token_end) in enumerate(tokenizer_offset_mapping):
                if token_start >= word_start - 1 and token_end <= word_end:
                    word_pos.append(i)
            token_positions.append(word_pos[0])
        # add the last token position
        token_positions.append(len(tokenizer_offset_mapping))

        if unit == 1:
            return token_positions
        else:
            _token_positions = []
            for i in range(0, len(token_positions), unit):
                _token_positions.append(token_positions[i])
            # add the last token position if smaller than the unit length
            if _token_positions[-1] < token_positions[-1]:
                _token_positions.append(token_positions[-1])
            return _token_positions
        

    def score(
            self,
            input: str,
            seq_len: int,
            n_sets: int,
            n_samples_per_set: int,
            forecast_horizons: List[int],
            add_bos_token: Optional[bool] = True,
            return_tokens: Optional[bool] = False,
            print_alternatives: Optional[bool] = False
    ):

        # check if seq_len is valid
        if type(seq_len) != int or seq_len < 1:
            raise ValueError("seq_len must be a positive integer")

        max_new_tokens = max(forecast_horizons) * seq_len

        # tokenize input string
        input_tokenized = self.tokenizer(input, return_offsets_mapping=True, return_attention_mask=False)

        # add special BOS token to obtain score of first input token
        if add_bos_token:
            input_tokenized['input_ids'] = [self.tokenizer.bos_token_id] + input_tokenized['input_ids']

        # if input sequences longer than context window, raise NotImplementedError
        if len(input_tokenized["input_ids"]) > self.max_seq_len:
            raise NotImplementedError(
                "Input sequence longer than model's context window. Truncation not yet implemented.")

        # map words to token positions
        timesteps = self.map_words_to_token_positions(input, seq_len, input_tokenized['offset_mapping'], skip_first_token=not add_bos_token)

        if add_bos_token:
            timesteps = [t + 1 for t in timesteps]
        # else:
        #     timesteps = timesteps[1:]
        # timesteps = [t + 1 for t in timesteps]

        # to tensor
        input_ids = torch.tensor(input_tokenized["input_ids"], device=self.device)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # list of lists of alternatives [t, n_samples_per_set * n_sets]
        alternatives = []
        lengths_true_seq = []

        for i, t in enumerate(timesteps):
            context_ids = input_ids[:t]

            alternatives_at_t = []

            length_true_seq = t - timesteps[i - 1] if i > 0 else t
            lengths_true_seq.append(length_true_seq)

            # for each sets of alternatives...
            for i in range(n_sets):
                # sample n_samples_per_set alternatives
                alternative_ids = self.model.generate(
                    context_ids.unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    num_return_sequences=n_samples_per_set,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                # store each sample in the list of alternatives at t
                # keep the last token of each context (the previous next-word target)
                for sample in alternative_ids.sequences:
                    alternative = sample[context_ids.shape[-1] - (length_true_seq):]
                    alternatives_at_t.append(alternative)
            
            if print_alternatives:
                print(f"Generated alternatives at timestep {t} ({self.tokenizer.decode(context_ids)}):")
                for i, alt in enumerate(alternatives_at_t):
                    print(f"{i}: {self.tokenizer.decode(alt)}")
                print('----------------------------------------------')


            # we have now collected all alternatives for this time step
            alternatives.append(alternatives_at_t)

        distances = defaultdict(list)
        for A_t, A_t_, length_true_seq_t, length_true_seq_t_ in zip(
                alternatives[:-1], alternatives[1:], lengths_true_seq[:-1], lengths_true_seq[1:]
        ):

            embeds_A, embeds_B = self.embed_seq_alternatives(A_t, A_t_, forecast_horizons, seq_len, length_true_seq_t,
                                                             length_true_seq_t_)

            for layer, horizon in embeds_A.keys():
                if self.standardize_embeddings:
                    embeds_A[(layer, horizon)] = (embeds_A[(layer, horizon)][0] - self.mean_std_embeds[(layer, horizon)][0]) / self.mean_std_embeds[(layer, horizon)][1]
                    embeds_B[(layer, horizon)] = (embeds_B[(layer, horizon)][0] - self.mean_std_embeds[(layer, horizon)][0]) / self.mean_std_embeds[(layer, horizon)][1]

                distances[(layer, horizon)].append(
                    self.pairwise_distances(
                        embeds_A[(layer, horizon)], embeds_B[(layer, horizon)]
                    )
                )

        rdict = defaultdict(list)
        for layer, horizon in distances.keys():
            for t in range(len(distances[layer, horizon])):
                for summary in distances[layer, horizon][t].keys():
                    rdict[f"forecast_{horizon}_layer_{layer}_summary_{summary}"].append(
                        distances[layer, horizon][t][summary].item()
                    )

        for k in rdict:
            if type(rdict[k]) == torch.Tensor:
                rdict[k] = rdict[k].numpy()

        if return_tokens:
            next_tokens = []
            for t, t_ in zip(timesteps[:-1], timesteps[1:]):
                next_tokens.append(
                    self.tokenizer.decode(
                        input_tokenized["input_ids"][t: t_]
                    ).strip()
                )
            rdict['tokens'] = next_tokens

        return rdict
