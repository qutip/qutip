import pytest
from qutip import QobjEvo

def test_qobjevo_return_updates_args():
    """Test that _return correctly updates args"""
    qobj = QobjEvo([("sigmax()", lambda t, args: args["a"])], args={"a": 1})  # Initial args
    qobj_updated = qobj._return({"a": 2})  # Should update args

    assert qobj_updated.args["a"] == 2, "Args were not updated correctly"

