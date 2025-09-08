"""
Generate imbalanced data with SynthBioData.
"""
from synthbiodata import create_config, generate_sample_data


def generate_imbalanced_data(seed=123):
    imbalanced_sample = create_config(
        data_type="molecular-descriptors",
        n_samples=1000,
        positive_ratio=0.1,  # 10% active, 90% inactive
        imbalanced=True,
        random_state=seed
    )

    # Generate data
    df = generate_sample_data(config=imbalanced_sample)

    # Print synthetic data summary
    print(f"Total samples: {len(df)}")
    print(f"Positive ratio: {df['binds_target'].mean():.1%}")

    return df


if __name__ == "__main__":
    df_imbalanced = generate_imbalanced_data()
