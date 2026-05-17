# README — Calculate the T(y) and H(y)

## Overview

This project provides a Python implementation for computing the functions:

- \(T(y)\)
- \(H(y)\)

which are defined through minimization problems involving the cumulative distribution function (CDF) of the Student's \(t\)-distribution and the standard normal distribution.

The implementation follows the theorem structure directly and supports:

- Numerical evaluation of \(T(y)\) and \(H(y)\)
- Automatic case distinction depending on \(y\)
- Detailed diagnostic information
- Identification of the minimizing degree of freedom \(v\)
- Stable computation using SciPy

---

## Mathematical Background

The code evaluates expressions of the form

\[
2F_v\left(y\sqrt{\frac{v}{v-2}}\right)-1
\]

and

\[
2-2F_v\left(y\sqrt{\frac{v}{v-2}}\right)
\]

where:

- \(F_v(x)\) is the CDF of the Student's \(t\)-distribution.
- \(\Phi(x)\) is the standard normal CDF.

The program computes finite minimizations over admissible ranges determined by the function \(v_0(y)\).

---

# Complete Source Code

```python
from audioop import error

from scipy import special
import math

try:
    import mpmath as mp

    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


def Phi(x):
    """Standard normal CDF"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def v_0(y):
    """Calculate v_0(y) as defined in the theorem"""
    if abs(y - math.sqrt(3)) < 1e-10:
        return 1318.4

    a = y ** 2
    a2 = y ** 4
    a3 = y ** 6
    a4 = y ** 8

    C1 = 2 * a3 + 0.75 * a2

    C2 = (
        2.25 * C1
        + (25.0 / 32.0) * a4
        + 1.5 * (
            C1 ** (2 / 3) * y ** (8 / 3)
            + C1 ** (1 / 3) * y ** (16 / 3)
        )
    )

    C3 = (13.0 / 3.0) * a2 + 18 * a + 2 * C2 + 11

    V1 = max(100.0, 8 * a)

    V2 = max(
        V1,
        1.5 * a2,
        math.sqrt(2 * C1)
    )

    V3 = max(
        V2,
        2 + 2 * a / (1 + 2 * y),
        2 + 2 * a2 / (1 + 2 * a)
    )

    v0 = max(
        V3,
        2 * C3 / abs(a - 3) + 1
    )

    return v0


def F(v, x):
    """Student's t-distribution CDF"""

    if v <= 2:
        raise ValueError(f"v must be > 2, got {v}")

    try:
        return special.stdtr(v, x)

    except:

        if v > 1000:
            return Phi(x)

        raise


def compute_term(v, y, mode='T'):
    """
    Compute theorem term.

    mode='T':
        2F_v(...) - 1

    mode='H':
        2 - 2F_v(...)
    """

    x_val = y * math.sqrt(v / (v - 2))

    f_val = F(v, x_val)

    if mode == 'T':
        return 2 * f_val - 1

    return 2 - 2 * f_val


def F_T(y, v_max_floor):
    """
    Compute

    F_T(y)
      = min_{3 <= v <= floor(v0)+3}
        {2F_v(...) - 1}
    """

    min_val = float('inf')
    min_v = None

    for v in range(3, v_max_floor + 1):

        try:
            val = compute_term(v, y, 'T')

            if val < min_val:
                min_val = val
                min_v = v

        except:
            continue

    return min_val, min_v


def F_H(y, v_max_floor):
    """
    Compute

    F_H(y)
      = min_{3 <= v <= floor(v0)+3}
        {2 - 2F_v(...)}
    """

    min_val = float('inf')
    min_v = None

    for v in range(3, v_max_floor + 1):

        try:
            val = compute_term(v, y, 'H')

            if val < min_val:
                min_val = val
                min_v = v

        except:
            continue

    return min_val, min_v


def T(y):
    """
    Compute T(y)
    """

    if y <= 0:
        raise ValueError("y must be > 0")

    v0 = v_0(y)

    v_max_floor = int(math.floor(v0)) + 3

    phi_term = 2 * Phi(y) - 1

    if y <= 1:

        return phi_term

    elif 1 < y < math.sqrt(3):

        f_t_val, _ = F_T(y, v_max_floor)

        return min(f_t_val, phi_term)

    elif y > math.sqrt(3):

        f_t_val, _ = F_T(y, v_max_floor)

        return f_t_val

    else:

        f_t_val, _ = F_T(y, v_max_floor)

        phi_term_at_sqrt3 = 2 * Phi(math.sqrt(3)) - 1

        return min(f_t_val, phi_term_at_sqrt3)


def H(y):
    """
    Compute H(y)
    """

    if y <= 0:
        raise ValueError("y must be > 0")

    v0 = v_0(y)

    v_max_floor = int(math.floor(v0)) + 3

    phi_term = 2 - 2 * Phi(y)

    if y <= 1:

        min_val = float('inf')

        for v in [3, 4]:

            try:
                val = compute_term(v, y, 'H')

                min_val = min(min_val, val)

            except:
                continue

        return min_val

    elif 1 < y < math.sqrt(3):

        f_h_val, _ = F_H(y, v_max_floor)

        return f_h_val

    elif y > math.sqrt(3):

        f_h_val, _ = F_H(y, v_max_floor)

        return min(f_h_val, phi_term)

    else:

        f_h_val, _ = F_H(y, v_max_floor)

        return f_h_val


def T_with_info(y):
    """
    Detailed information for T(y)
    """

    if y <= 0:
        raise ValueError("y must be greater than 0")

    v0 = v_0(y)

    v_max_floor = int(math.floor(v0)) + 3

    phi_term = 2 * Phi(y) - 1

    result = {
        'y': y,
        'v0': v0,
        'v_max': v_max_floor,
        'finite_terms': [],
        'phi_term': phi_term,
        'min_value': None,
        'min_v': None,
        'case': None
    }

    if y <= 1:

        result['min_value'] = phi_term
        result['min_v'] = 'infinity'
        result['case'] = '0 < y <= 1'

        return result

    for v in range(3, v_max_floor + 1):

        try:
            val = compute_term(v, y, 'T')

            result['finite_terms'].append((v, val))

        except:
            continue

    if result['finite_terms']:

        result['finite_terms'].sort(
            key=lambda x: x[1]
        )

        min_finite_v, min_finite_val = (
            result['finite_terms'][0]
        )

    else:

        min_finite_v = float('inf')
        min_finite_val = None

    if 1 < y < math.sqrt(3):

        result['case'] = '1 < y < sqrt(3)'

        if min_finite_val < phi_term:

            result['min_value'] = min_finite_val
            result['min_v'] = min_finite_v

        else:

            result['min_value'] = phi_term
            result['min_v'] = 'infinity'

    elif y > math.sqrt(3):

        result['case'] = 'y > sqrt(3)'

        result['min_value'] = min_finite_val
        result['min_v'] = min_finite_v

    else:

        result['case'] = 'y = sqrt(3)'

        phi_at_sqrt3 = (
            2 * Phi(math.sqrt(3)) - 1
        )

        result['min_value'] = min(
            min_finite_val,
            phi_at_sqrt3
        )

        if min_finite_val < phi_at_sqrt3:

            result['min_v'] = min_finite_v

        else:

            result['min_v'] = 'infinity'

    return result


def H_with_info(y):
    """
    Detailed information for H(y)
    """

    if y <= 0:
        raise ValueError("y must be greater than 0")

    v0 = v_0(y)

    v_max_floor = int(math.floor(v0)) + 3

    phi_term = 2 - 2 * Phi(y)

    result = {
        'y': y,
        'v0': v0,
        'v_max': v_max_floor,
        'finite_terms': [],
        'phi_term': phi_term,
        'min_value': None,
        'min_v': None,
        'case': None
    }

    if y <= 1:

        result['case'] = '0 < y <= 1'

        min_val = float('inf')
        min_v = None

        for v in [3, 4]:

            try:
                val = compute_term(v, y, 'H')

                result['finite_terms'].append(
                    (v, val)
                )

                if val < min_val:
                    min_val = val
                    min_v = v

            except:
                continue

        result['min_value'] = min_val
        result['min_v'] = min_v

        return result

    for v in range(3, v_max_floor + 1):

        try:
            val = compute_term(v, y, 'H')

            result['finite_terms'].append(
                (v, val)
            )

        except:
            continue

    if result['finite_terms']:

        result['finite_terms'].sort(
            key=lambda x: x[1]
        )

        min_finite_v, min_finite_val = (
            result['finite_terms'][0]
        )

    else:

        min_finite_v = float('inf')
        min_finite_val = None

    if 1 < y < math.sqrt(3):

        result['case'] = '1 < y < sqrt(3)'

        result['min_value'] = min_finite_val
        result['min_v'] = min_finite_v

    elif y > math.sqrt(3):

        result['case'] = 'y > sqrt(3)'

        if min_finite_val < phi_term:

            result['min_value'] = min_finite_val
            result['min_v'] = min_finite_v

        else:

            result['min_value'] = phi_term
            result['min_v'] = 'infinity'

    else:

        result['case'] = 'y = sqrt(3)'

        result['min_value'] = min_finite_val
        result['min_v'] = min_finite_v

    return result


def format_number(x):
    """
    Format number
    """

    if (
        x == float('inf')
        or x == -float('inf')
        or math.isnan(x)
    ):
        return str(x)

    return f"{x:.15g}"


def print_results(t_info, h_info):
    """
    Pretty formatted printing
    """

    y = t_info['y']

    print(f"y = {y}")

    if y < 0:

        error

    elif 0 < y <= 1:

        print(
            "This case belongs to y in (0,1]"
        )

    else:

        print(
            f"v0 = {t_info['v0']:.2f}, "
            f"v_max = {t_info['v_max']}"
        )

    print()

    print(f"T({y}):")

    print(f"  Case: {t_info['case']}")

    if t_info['finite_terms']:

        print("  Top 5 finite terms:")

        for v, val in t_info['finite_terms'][:5]:

            print(
                f"    v={v}: "
                f"{format_number(val)}"
            )

    print(
        f"  Phi term: "
        f"{format_number(t_info['phi_term'])}"
    )

    print(
        f"  Result: "
        f"T = {format_number(t_info['min_value'])} "
        f"at v = {t_info['min_v']}"
    )

    print()

    print(f"H({y}):")

    print(f"  Case: {h_info['case']}")

    if h_info['finite_terms']:

        print("  Top 5 finite terms:")

        for v, val in h_info['finite_terms'][:5]:

            print(
                f"    v={v}: "
                f"{format_number(val)}"
            )

    print(
        f"  Phi term: "
        f"{format_number(h_info['phi_term'])}"
    )

    print(
        f"  Result: "
        f"H = {format_number(h_info['min_value'])} "
        f"at v = {h_info['min_v']}"
    )

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":

    test_values = [
        1.0,
        math.sqrt(3),
        2.0,
        3.0
    ]

    for y in test_values:

        try:

            t_info = T_with_info(y)

            h_info = H_with_info(y)

            print_results(
                t_info,
                h_info
            )

        except Exception as e:

            print(f"Error at y={y}: {e}")
```

---

# Installation

Install required dependencies:

```bash
pip install scipy
```

Optional:

```bash
pip install mpmath
```

---

# Usage

Run the script directly:

```bash
python "Calculate the T(y) and H(y).py"
```

---

# Example

```python
from math import sqrt
from your_file_name import T, H

y = 2.0

print(T(y))
print(H(y))
```

---

# Example Detailed Output

```python
from your_file_name import (
    T_with_info,
    H_with_info,
    print_results
)

y = 2.0

t_info = T_with_info(y)

h_info = H_with_info(y)

print_results(t_info, h_info)
```

---

# Mathematical Definitions

## Definition of \(T(y)\)

\[
T(y)=
\begin{cases}
2\Phi(y)-1,
& 0<y\le1
\\
\min\{F_T(y),\,2\Phi(y)-1\},
& 1<y<\sqrt3
\\
F_T(y),
& y>\sqrt3
\end{cases}
\]

where

\[
F_T(y)=
\min_{3\le v\le \lfloor v_0(y)\rfloor+3}
\left\{
2F_v\left(
y\sqrt{\frac{v}{v-2}}
\right)-1
\right\}
\]

---

## Definition of \(H(y)\)

\[
H(y)=
\begin{cases}
\min_{v=3,4}
\left\{
2-2F_v\left(
y\sqrt{\frac{v}{v-2}}
\right)
\right\},
& 0<y\le1
\\
F_H(y),
& 1<y<\sqrt3
\\
\min\{F_H(y),\,2-2\Phi(y)\},
& y>\sqrt3
\end{cases}
\]

where

\[
F_H(y)=
\min_{3\le v\le \lfloor v_0(y)\rfloor+3}
\left\{
2-2F_v\left(
y\sqrt{\frac{v}{v-2}}
\right)
\right\}
\]

---

# Numerical Notes

- Student's \(t\)-CDF is computed using:

```python
scipy.special.stdtr
```

- For very large \(v\), the implementation uses the normal approximation:

\[
F_v(x)\approx \Phi(x)
\]

- Numerical safeguards are included for \(v \le 2\).

---

# Future Improvements

Possible extensions include:

- NumPy vectorization
- Parallel minimization
- Plotting \(T(y)\) and \(H(y)\)
- Higher precision arithmetic
- Jupyter notebook demos
- LaTeX export support

---

# License

Academic and research use only.

            print(f"Error at y={y}: {e}")
