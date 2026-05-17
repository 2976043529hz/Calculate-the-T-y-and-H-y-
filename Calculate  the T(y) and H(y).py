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
        return 1318.4  # \bar{v}_0(\sqrt{3})

    a = y ** 2
    a2 = y ** 4
    a3 = y ** 6
    a4 = y ** 8

    C1 = 2 * a3 + 0.75 * a2
    C2 = 2.25 * C1 + (25.0 / 32.0) * a4 + 1.5 * (C1 ** (2 / 3) * y ** (8 / 3) + C1 ** (1 / 3) * y ** (16 / 3))
    C3 = (13.0 / 3.0) * a2 + 18 * a + 2 * C2 + 11

    V1 = max(100.0, 8 * a)
    V2 = max(V1, 1.5 * a2, math.sqrt(2 * C1))
    V3 = max(V2, 2 + 2 * a / (1 + 2 * y), 2 + 2 * a2 / (1 + 2 * a))

    v0 = max(V3, 2 * C3 / abs(a - 3) + 1)
    return v0


def F(v, x):
    """Student's t CDF with numerical stability"""
    if v <= 2:
        raise ValueError(f"v must be > 2, got {v}")

    try:
        # Use scipy's implementation
        return special.stdtr(v, x)
    except:
        # Fallback: normal approximation for large v
        if v > 1000:
            return Phi(x)
        raise


def compute_term(v, y, mode='T'):
    """
    Compute the term for a given v
    mode: 'T' for T(y) term: 2F_v(...) - 1
          'H' for H(y) term: 2 - 2F_v(...)
    """
    x_val = y * math.sqrt(v / (v - 2))
    f_val = F(v, x_val)

    if mode == 'T':
        return 2 * f_val - 1
    else:  # 'H'
        return 2 - 2 * f_val


def F_T(y, v_max_floor):
    """Compute F_T(y) = min_{3 ≤ v ≤ floor(v0)+3} {2F_v(...)-1}"""
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
    """Compute F_H(y) = min_{3 ≤ v ≤ floor(v0)+3} {2-2F_v(...)}"""
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
    Compute T(y) as defined in Theorem (i)
    """
    if y <= 0:
        raise ValueError("y must be > 0")

    a = y ** 2
    v0 = v_0(y)
    v_max_floor = int(math.floor(v0)) + 3

    phi_term = 2 * Phi(y) - 1

    if y <= 1:  # 0 < y ≤ 1
        return phi_term

    elif 1 < y < math.sqrt(3):  # 1 < y < √3
        f_t_val, _ = F_T(y, v_max_floor)
        return min(f_t_val, phi_term)

    elif y > math.sqrt(3):  # y > √3
        f_t_val, _ = F_T(y, v_max_floor)
        return f_t_val

    else:  # y == √3
        f_t_val, _ = F_T(y, v_max_floor)
        phi_term_at_sqrt3 = 2 * Phi(math.sqrt(3)) - 1
        return min(f_t_val, phi_term_at_sqrt3)


def H(y):
    """
    Compute H(y) as defined in Theorem (ii)
    """
    if y <= 0:
        raise ValueError("y must be > 0")

    a = y ** 2
    v0 = v_0(y)
    v_max_floor = int(math.floor(v0)) + 3

    phi_term = 2 - 2 * Phi(y)

    if y <= 1:  # 0 < y ≤ 1
        # min_{v=3,4}
        min_val = float('inf')
        for v in [3, 4]:
            try:
                val = compute_term(v, y, 'H')
                min_val = min(min_val, val)
            except:
                continue
        return min_val

    elif 1 < y < math.sqrt(3):  # 1 < y < √3
        f_h_val, _ = F_H(y, v_max_floor)
        return f_h_val

    elif y > math.sqrt(3):  # y > √3
        f_h_val, _ = F_H(y, v_max_floor)
        return min(f_h_val, phi_term)

    else:  # y == √3
        f_h_val, _ = F_H(y, v_max_floor)
        return f_h_val


def T_with_info(y):
    """T(y) with detailed information"""
    if y <= 0:
        raise ValueError("y must be greater than 0")

    a = y ** 2
    v0 = v_0(y)
    v_max_floor = int(math.floor(v0)) + 3
    phi_term = 2 * Phi(y) - 1

    result = {
        'y': y,
        'y_squared': a,
        'v0': v0,
        'v_max': v_max_floor,
        'finite_terms': [],
        'phi_term': phi_term,
        'min_value': None,
        'min_v': None,
        'case': None
    }

    # Compute finite terms for all needed v
    start_v = 3
    end_v = v_max_floor

    # Special case: y ≤ 1 doesn't need finite terms
    if y <= 1:
        result['min_value'] = phi_term
        result['min_v'] = 'infinity'
        result['case'] = '0 < y ≤ 1'
        return result

    for v in range(start_v, end_v+1):
        try:
            val = compute_term(v, y, 'T')
            result['finite_terms'].append((v, val))
        except:
            continue

    if result['finite_terms']:
        result['finite_terms'].sort(key=lambda x: x[1])
        min_finite_v, min_finite_val = result['finite_terms'][0]
    else:
        min_finite_v, min_finite_val = float('inf'), None

    # Apply theorem logic
    if 1 < y < math.sqrt(3):
        result['case'] = '1 < y < √3'
        if min_finite_val < phi_term:
            result['min_value'] = min_finite_val
            result['min_v'] = min_finite_v
        else:
            result['min_value'] = phi_term
            result['min_v'] = 'infinity'
    elif y > math.sqrt(3):
        result['case'] = 'y > √3'
        result['min_value'] = min_finite_val
        result['min_v'] = min_finite_v
    else:  # y == √3
        result['case'] = 'y = √3'
        phi_at_sqrt3 = 2 * Phi(math.sqrt(3)) - 1
        result['min_value'] = min(min_finite_val, phi_at_sqrt3)
        result['min_v'] = min_finite_v if min_finite_val < phi_at_sqrt3 else 'infinity'

    return result


def H_with_info(y):
    """H(y) with detailed information"""
    if y <= 0:
        raise ValueError("y must be greater than 0")

    a = y ** 2
    v0 = v_0(y)
    v_max_floor = int(math.floor(v0)) + 3
    phi_term = 2 - 2 * Phi(y)

    result = {
        'y': y,
        'y_squared': a,
        'v0': v0,
        'v_max': v_max_floor,
        'finite_terms': [],
        'phi_term': phi_term,
        'min_value': None,
        'min_v': None,
        'case': None
    }

    # Special case: y ≤ 1 only needs v=3,4
    if y <= 1:
        result['case'] = '0 < y ≤ 1'
        min_val = float('inf')
        min_v = None
        for v in [3, 4]:
            try:
                val = compute_term(v, y, 'H')
                result['finite_terms'].append((v, val))
                if val < min_val:
                    min_val = val
                    min_v = v
            except:
                continue
        result['min_value'] = min_val
        result['min_v'] = min_v
        return result

    # Compute finite terms
    for v in range(3, v_max_floor+1):
        try:
            val = compute_term(v, y, 'H')
            result['finite_terms'].append((v, val))
        except:
            continue

    if result['finite_terms']:
        result['finite_terms'].sort(key=lambda x: x[1])
        min_finite_v, min_finite_val = result['finite_terms'][0]
    else:
        min_finite_v, min_finite_val = float('inf'), None

    # Apply theorem logic
    if 1 < y < math.sqrt(3):
        result['case'] = '1 < y < √3'
        result['min_value'] = min_finite_val
        result['min_v'] = min_finite_v
    elif y > math.sqrt(3):
        result['case'] = 'y > √3'
        if min_finite_val < phi_term:
            result['min_value'] = min_finite_val
            result['min_v'] = min_finite_v
        else:
            result['min_value'] = phi_term
            result['min_v'] = 'infinity'
    else:  # y == √3
        result['case'] = 'y = √3'
        result['min_value'] = min_finite_val
        result['min_v'] = min_finite_v

    return result


def format_number(x):
    """Format number with 15 significant digits"""
    if x == float('inf') or x == -float('inf') or math.isnan(x):
        return str(x)
    return f"{x:.15g}"


def print_results(t_info, h_info):
    """Print both results in formatted way"""
    y = t_info['y']

    print(f"y = {y}")
    if y<0:
        error
    elif 0<y<=1:
        print('this case is belong to the case y in (0,1]')
    else:
        print(f"v0 = {t_info['v0']:.2f}, v_max = {t_info['v_max']}")

    print()

    # T(y) results
    print(f"T({y}):")
    print(f"  Case: {t_info['case']}")
    if t_info['finite_terms']:
        print(f"  Top 5 finite terms (v, T):")
        for v, val in t_info['finite_terms'][:5]:
            print(f"    v={v}: {format_number(val)}")
    print(f"  Phi term: {format_number(t_info['phi_term'])}")
    print(f"  Result: T = {format_number(t_info['min_value'])} at v = {t_info['min_v']}")
    print()

    # H(y) results
    print(f"H({y}):")
    print(f"  Case: {h_info['case']}")
    if h_info['finite_terms']:
        print(f"  Top 5 finite terms (v, H):")
        for v, val in h_info['finite_terms'][:5]:
            print(f"    v={v}: {format_number(val)}")
    print(f"  Phi term: {format_number(h_info['phi_term'])}")
    print(f"  Result: H = {format_number(h_info['min_value'])} at v = {h_info['min_v']}")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    test_values = [ 1.0, math.sqrt(3), 2.0, 3.0]

    for y in test_values:
        try:
            t_info = T_with_info(y)
            h_info = H_with_info(y)
            print_results(t_info, h_info)
        except Exception as e:
            print(f"Error at y={y}: {e}")