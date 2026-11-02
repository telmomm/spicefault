# The cheapest certified plan

The Sallen-Key band-pass of the test design example as the self-test of a board: fault
rates from public tables, the candidate tests read through the ADC of an STM32G474
with four free channels, the plan of least hardware cost whose DC_w reaches 0.97 by
integer programming and by the greedy, its coverage on circuits that did not choose
it, the claim certified on new circuits, and the ambiguity groups the plan gives up.

```bash
pip install -e ".[ilp]"
python examples/10_minimum_cost_plan.py
```

```python
--8<-- "examples/10_minimum_cost_plan.py"
```
