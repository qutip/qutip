Building QuTiP with 64-bit Integer Indices
===========================================

By default, QuTiP uses 32-bit integer indices, which is sufficient for most cases. However, for users working with **very large** quantum systems, QuTiP supports **64-bit integer indices**.

### **How to Enable 64-bit Indices**
To build QuTiP with 64-bit indices, use the following command:

.. code-block:: bash

    python -m build \
        --wheel \
        --config-setting="--global-option=--with-idxint-64"

This ensures that QuTiP compiles with 64-bit integer indices instead of the default 32-bit.

### **Checking if 64-bit Indices Are Enabled**
To verify that your build uses 64-bit indices, run:

.. code-block:: python

    import qutip
    print(qutip.data.idxint_dtype)  # Should output 'int64'

### **When to Use 64-bit Indices?**
- If you need to handle **very large quantum objects** (e.g., large sparse matrices).
- If you encounter **index overflow errors** when working with large systems.

### **Limitations**
- 64-bit indices **increase memory usage**.
- Some operations **may run slightly slower** compared to 32-bit versions.

For more details, refer to the official documentation or open an issue on GitHub.
