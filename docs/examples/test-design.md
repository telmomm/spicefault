# Test design

The self-test of a Sallen-Key band-pass: 52 candidate tests over three stimuli, the
detection of 28 faults and the false alarm rate, the coverage metrics with bounds that
hold for all faults at once, the tests chosen by a published selection method, and a
stopping rule that simulates until the claim "DC_w ≥ 0.9" is decided.

```bash
python examples/09_test_design.py
```

```python
--8<-- "examples/09_test_design.py"
```
