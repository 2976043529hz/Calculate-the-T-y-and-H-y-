# README — Calculate the T(y) and H(y)

## Overview

This project provides a Python implementation for computing the functions:

- $T(y)$
- $H(y)$

which are defined through minimization problems involving the cumulative distribution function (CDF) of the Student's $t$-distribution and the standard normal distribution.

The implementation follows the theorem structure directly and supports:

- Numerical evaluation of $T(y)$ and $H(y)$
- Automatic case distinction depending on $y$
- Detailed diagnostic information
- Identification of the minimizing degree of freedom $v$
- Stable computation using SciPy

---

## Mathematical Background

The code evaluates expressions of the form

$$
2F_v\left(y\sqrt{\frac{v}{v-2}}\right)-1
$$

and

$$
2-2F_v\left(y\sqrt{\frac{v}{v-2}}\right)
$$

where:

- $F_v(x)$ is the CDF of the Student's $t$-distribution.
- $\Phi(x)$ is the standard normal CDF.

The program computes finite minimizations over admissible ranges determined by the function $v_0(y)$.

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
from your_file_name import T, H

y = 2.0

print(T(y))
print(H(y))
```

---

# Mathematical Definitions

## Definition of $T(y)$

$$
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
$$

where

$$
F_T(y)=
\min_{3\le v\le \lfloor v_0(y)\rfloor+3}
\{
2F_v(
y\sqrt{\frac{v}{v-2}}
)-1
\}
$$

---

## Definition of $H(y)$

$$
H(y)=
\begin{cases}
\min_{v=3,4}
\{
2-2F_v(
y\sqrt{\frac{v}{v-2}}
)
\},
& 0<y\le1
\\
F_H(y),
& 1<y<\sqrt3
\\
\min\{F_H(y),\,2-2\Phi(y)\},
& y>\sqrt3
\end{cases}
$$

where

$$
F_H(y)=
\min_{3\le v\le \lfloor v_0(y)\rfloor+3}
\{
2-2F_v(
y\sqrt{\frac{v}{v-2}}
)
\}
$$

---

# Numerical Notes

- Student's $t$-CDF is computed using:

```python
scipy.special.stdtr
```

- For very large $v$, the implementation uses the normal approximation:

$$
F_v(x)\approx \Phi(x)
$$

- Numerical safeguards are included for $v \le 2$.

---

# Future Improvements

Possible extensions include:

- NumPy vectorization
- Parallel minimization
- Plotting $T(y)$ and $H(y)$
- Higher precision arithmetic
- Jupyter notebook demos
- LaTeX export support

---

# License

Academic and research use only.
