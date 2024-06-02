import argparse
import pandas as pd

from measures import SurprisalScorer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute surprisal for a given dataset')
    parser.add_argument('--dataset', type=str, help='Path to the dataset')
    parser.add_argument('--output', type=str, help='Path to save the output')
    parser.add_argument('--aggregate_by_word', action='store_true', help='Aggregate surprisal by word')
    parser.add_argument('--return_tokens', action='store_true', help='Return tokens')
    parser.add_argument('--model_name_or_path', type=str, help='Model name or path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--debug_n', type=int, default=0, help='Number of documents to debug')
    args = parser.parse_args()

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
    scorer = SurprisalScorer(model_name_or_path=args.model_name_or_path, device=args.device)

    surprisals = []
    entropies = []
    deviations = []
    processed_ids = []
    tokens = []

    # Compute surprisal
    for id, text in list(zip(ids, texts)):
        surprisal_rdict = scorer.score(
            text,
            aggregate_by_word=args.aggregate_by_word,
            return_tokens=args.return_tokens,
        )
        surprisals.append(surprisal_rdict['surprisal'])
        entropies.append(surprisal_rdict['entropy'])
        deviations.append(surprisal_rdict['deviation'])
        processed_ids.append(id)
        if args.return_tokens:
            tokens.append(surprisal_rdict['tokens'])

    # Create a dataframe
    output_df = pd.DataFrame({
        'id': processed_ids,
        'surprisal': surprisals,
        'entropy': entropies,
        'deviation': deviations,
        'tokens': tokens if args.return_tokens else ''
    })

    # Save the dataframe
    output_df.to_csv(args.output, index=False)
