import argparse
import pandas as pd

from collections import defaultdict

from measures import IncrementalInformationValueScorer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute incremental information value for a given dataset')
    parser.add_argument('--dataset', type=str, help='Path to the dataset')
    parser.add_argument('--output', type=str, help='Path to save the output')
    parser.add_argument('--aggregate_by_word', action='store_true', help='Aggregate surprisal by word')
    parser.add_argument('--return_tokens', action='store_true', help='Return tokens')
    parser.add_argument('--model_name_or_path', type=str, help='Model name or path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--layers', type=str, default=None, help='Layers to use')
    parser.add_argument('--summary_fn', type=str, default=None, help='Summary function to use')
    parser.add_argument('--seq_len', type=int, default=1, help='Unit sequence length to use')
    parser.add_argument('--n_sets', type=int, default=1, help='Number of alternative sets to use')
    parser.add_argument('--n_samples_per_set', type=int, default=1, help='Number of samples per set to use')
    parser.add_argument('--forecast_horizons', type=str, default=None, help='Forecast horizons to use')
    parser.add_argument('--add_bos_token', action='store_true', help='Add BOS token')
    parser.add_argument('--seed', type=int, default=0, help='Seed to use')
    parser.add_argument('--debug_n', type=int, default=0, help='Number of documents to debug')
    parser.add_argument('--log_every_n', type=int, default=10, help='Log progress every n documents')
    args = parser.parse_args()

    args.layers = eval(args.layers) if args.layers else None
    args.forecast_horizons = eval(args.forecast_horizons) if args.forecast_horizons else None

    # Load csv dataset of documents using pandas
    dataset = pd.read_csv(args.dataset)
    assert 'text' in dataset.columns, 'The dataset should contain a column named "text"'
    assert 'id' in dataset.columns, 'The dataset should contain a column named "id"'

    texts = dataset['text'].tolist()
    ids = dataset['id'].tolist()
    if args.debug_n:
        texts = texts[:args.debug_n]
        ids = ids[:args.debug_n]

    # Initialize the scorer
    scorer = IncrementalInformationValueScorer(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        layers=args.layers,
        summary_fn=args.summary_fn,
        seed=args.seed
    )

    summary_fns = [args.summary_fn] if args.summary_fn else ['mean', 'min', 'max']

    iv_scores = defaultdict(list)
    tokens = defaultdict(list)

    # Compute incremental information value
    for i, text in enumerate(texts, start=1):

        # print progress every log_every_n documents
        if i % args.log_every_n == 0:
            print(f'Processing document {i} of {len(texts)}')

        rdict = scorer.score(
            text,
            seq_len=args.seq_len,
            n_sets=args.n_sets,
            n_samples_per_set=args.n_samples_per_set,
            forecast_horizons=args.forecast_horizons,
            add_bos_token=args.add_bos_token,
            return_tokens=args.return_tokens,
        )

        for horizon in args.forecast_horizons:
            for layer in args.layers:
                for summary in summary_fns:
                    iv_scores[(horizon, layer, summary)].append(
                        rdict[f'forecast_{horizon}_layer_{layer}_summary_{summary}']
                    )
                    if args.return_tokens:
                        tokens[(horizon, layer, summary)].append(
                            rdict['tokens']
                        )

    # Create output dataframe
    df_content = []

    for horizon in args.forecast_horizons:
        for layer in args.layers:
            for summary in summary_fns:
                _scores = iv_scores[(horizon, layer, summary)]
                _tokens = tokens[(horizon, layer, summary)]

                for text_id in ids:
                    row_dict = {
                        'id': text_id,
                        'horizon': horizon,
                        'layer': layer,
                        'summary': summary,
                        'score': _scores.pop(0),
                    }
                    if args.return_tokens:
                        row_dict['tokens'] = _tokens.pop(0)
                    df_content.append(row_dict)

                assert len(_scores) == 0, 'There are still scores left'
                assert len(_tokens) == 0, 'There are still tokens left'

    output_df = pd.DataFrame(df_content)
    output_df.to_csv(args.output, index=False)
