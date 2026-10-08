import pandas as pd, pytest
from mk12.submit import validate
from mk12.data import load


def test_validator_rejects_broken():
    te = load('test')
    good = pd.DataFrame({'trace_id': te.trace_id, 'root_code': 'NONE'})
    validate(good)
    with pytest.raises(AssertionError):
        validate(good.iloc[:-1])
    bad = good.copy(); bad.loc[0, 'root_code'] = 'FOO-99'
    with pytest.raises(AssertionError):
        validate(bad)
