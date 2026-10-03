"""The dataset of a campaign: integrity, provenance and reproduction.

    python examples/04_dataset.py

It uses the dataset written by `03_campaign.py`, and runs that campaign if needed.
"""

from rc_study import WORKERS, campaign

from spicefault import Dataset

if __name__ == "__main__":
    campaign().run(workers=WORKERS, chunk=200, progress=False)

    # everything below needs only the folder
    dataset = Dataset(campaign().out_dir)
    print(dataset)
    print("files:", sorted(p.name for p in dataset.path.iterdir()))
    print("problems found by verify():", dataset.verify())

    sample_id = 350
    print(f"\nprovenance of sample {sample_id}:")
    for key, value in dataset.provenance(sample_id).to_dict().items():
        print(f"  {key}: {value}")

    print(f"\nthe netlist that was simulated for sample {sample_id}:")
    print(dataset.netlist(sample_id))

    print("simulating 20 samples again and comparing with what is stored:")
    report = dataset.reproduce(n=20)
    print(report.head().to_string(index=False))
    print("largest difference in the measurements:", report["max_abs_diff"].max())
    print("largest difference in the waveforms:   ", report["waveform_abs_diff"].max())

    x, y = dataset.to_ml(target="fault_location")
    print(f"\nfor a classifier: X {x.shape}, labels {sorted(set(y))}")
    waves, kind = dataset.to_ml(target="fault_type", waveforms=True)
    print(f"or the waveforms: X {waves.shape}, labels {sorted(set(kind))}")
