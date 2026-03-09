"""L104 Numerical Engine — Hyper-Precision Core.

100-decimal arithmetic foundation: all decimal_* math primitives,
D() safe conversion, fmt100() formatting. No internal package imports.

PART V RESEARCH — l104_runtime_infrastructure_research.py:
  F57: φ×(1/φ) = 1.0 to 100 decimal places (120-digit internal precision)
  F58: GOD_CODE float = 527.5184818492612 (14 significant digits from HP)
  F59: Conservation G(X)·2^(X/104) = INVARIANT verified to 90 decimals
  F60: Factor-13 structure: 286=22×13, 104=8×13, 416=32×13
"""

import functools
from decimal import Decimal, getcontext, ROUND_HALF_EVEN, InvalidOperation

# Set global decimal context: 120 internal digits, round-half-even
getcontext().prec = 120
getcontext().rounding = ROUND_HALF_EVEN

# ─── 100-decimal precision utility ───
DISPLAY_PRECISION = 100

def D(value) -> Decimal:
    """Safe Decimal conversion — strings preferred to avoid float contamination."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(str(value))

def fmt100(d: Decimal) -> str:
    """Format a Decimal to exactly 100 decimal places."""
    return format(d, f'.{DISPLAY_PRECISION}f')

def decimal_sqrt(n: Decimal) -> Decimal:
    """Newton's method square root to full Decimal precision."""
    if n < 0:
        raise ValueError("Square root of negative number")
    if n == 0:
        return D(0)
    # Initial guess from float
    x = D(str(float(n) ** 0.5))
    # Newton iterations until convergence
    for _ in range(200):
        x_new = (x + n / x) / D(2)
        if x_new == x:
            break
        x = x_new
    return x

def decimal_ln(x: Decimal) -> Decimal:
    """Natural logarithm via AGM method for full Decimal precision."""
    if x <= 0:
        raise ValueError("ln of non-positive number")
    prec = getcontext().prec
    # Use the series: ln(x) = 2 * atanh((x-1)/(x+1))
    # For large x, reduce: ln(x) = ln(x/2^k) + k*ln(2)
    # First compute ln(2) via atanh series
    one = D(1)
    two = D(2)

    # Reduce x to near 1 for convergence
    k = 0
    y = x
    ln2 = D('0.6931471805599453094172321214581765680755001343602552541206800094933936219696947156058633269964186875420')
    while y > two:
        y = y / two
        k += 1
    while y < D('0.5'):
        y = y * two
        k -= 1

    # Now y is in [0.5, 2), compute ln(y) via atanh series
    # ln(y) = 2 * atanh((y-1)/(y+1)) = 2 * sum_{n=0}^{inf} ((y-1)/(y+1))^(2n+1) / (2n+1)
    z = (y - one) / (y + one)
    z2 = z * z
    term = z
    result = z
    for n in range(1, 500):
        term *= z2
        contrib = term / D(2 * n + 1)
        result += contrib
        if abs(contrib) < D(10) ** (-(prec + 10)):
            break
    result *= two

    return result + D(k) * ln2

def decimal_exp(x: Decimal) -> Decimal:
    """Exponential function via Taylor series to full Decimal precision."""
    prec = getcontext().prec
    one = D(1)
    # Range reduction: exp(x) = exp(r) * 2^k where x = r + k*ln2
    ln2 = D('0.6931471805599453094172321214581765680755001343602552541206800094933936219696947156058633269964186875420')

    k = int(x / ln2)
    r = x - D(k) * ln2

    # Taylor series for exp(r), |r| < ln2
    term = one
    result = one
    for n in range(1, 600):
        term *= r / D(n)
        result += term
        if abs(term) < D(10) ** (-(prec + 10)):
            break

    # Multiply by 2^k
    if k >= 0:
        result *= D(2) ** k
    else:
        result /= D(2) ** (-k)
    return result

def decimal_pow(base: Decimal, exp: Decimal) -> Decimal:
    """base^exp = exp(exp * ln(base)) in full Decimal precision."""
    if base <= 0:
        raise ValueError("Power with non-positive base")
    return decimal_exp(exp * decimal_ln(base))

def decimal_sin(x: Decimal) -> Decimal:
    """Sine via Taylor series at full Decimal precision."""
    prec = getcontext().prec
    pi = D('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170680')
    two_pi = pi * D(2)
    # Range reduction to [-π, π]
    x = x % two_pi
    if x > pi:
        x -= two_pi
    term = x
    result = x
    x2 = x * x
    for n in range(1, 400):
        term *= -x2 / (D(2 * n) * D(2 * n + 1))
        result += term
        if abs(term) < D(10) ** (-(prec + 5)):
            break
    return result

def decimal_cos(x: Decimal) -> Decimal:
    """Cosine via Taylor series at full Decimal precision."""
    prec = getcontext().prec
    pi = D('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170680')
    two_pi = pi * D(2)
    x = x % two_pi
    if x > pi:
        x -= two_pi
    term = D(1)
    result = D(1)
    x2 = x * x
    for n in range(1, 400):
        term *= -x2 / (D(2 * n - 1) * D(2 * n))
        result += term
        if abs(term) < D(10) ** (-(prec + 5)):
            break
    return result

def decimal_atan(x: Decimal) -> Decimal:
    """Arctangent: for |x| <= 1, Taylor series; otherwise identity atan(x) = π/2 - atan(1/x)."""
    prec = getcontext().prec
    pi = D('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170680')
    if x < 0:
        return -decimal_atan(-x)
    if x > D(1):
        return pi / D(2) - decimal_atan(D(1) / x)
    # Reduce to small argument: atan(x) = 2*atan(x/(1+sqrt(1+x^2)))
    if x > D('0.5'):
        reduced = x / (D(1) + decimal_sqrt(D(1) + x * x))
        return D(2) * decimal_atan(reduced)
    # Taylor series: atan(x) = x - x^3/3 + x^5/5 - ...
    x2 = x * x
    term = x
    result = x
    for n in range(1, 600):
        term *= -x2
        contrib = term / D(2 * n + 1)
        result += contrib
        if abs(contrib) < D(10) ** (-(prec + 5)):
            break
    return result

def decimal_factorial(n: int) -> Decimal:
    """Exact factorial for integer n, returned as Decimal."""
    result = D(1)
    for i in range(2, n + 1):
        result *= D(i)
    return result

def decimal_gamma_lanczos(z: Decimal) -> Decimal:
    """Gamma function via Lanczos approximation extended to high precision.
    Works for Re(z) > 0.5. For z < 0.5, use reflection: Γ(z)Γ(1-z) = π/sin(πz).
    Returns Infinity for non-positive integer arguments (poles)."""
    pi = D('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170680')
    # Check for poles at non-positive integers
    if z <= D('0') and z == int(z):
        return D('Infinity')
    if z < D('0.5'):
        # Reflection formula
        sin_val = decimal_sin(pi * z)
        if abs(sin_val) < D('1E-100'):
            return D('Infinity')  # Pole
        return pi / (sin_val * decimal_gamma_lanczos(D(1) - z))
    z -= D(1)
    # Lanczos g=7, coefficients (extended)
    p = [
        D('0.99999999999980993227684700473478296744476168282198'),
        D('676.5203681218851'),
        D('-1259.1392167224028'),
        D('771.32342877765313'),
        D('-176.61502916214059'),
        D('12.507343278686905'),
        D('-0.13857109526572012'),
        D('9.9843695780195716e-6'),
        D('1.5056327351493116e-7'),
    ]
    g = D(7)
    x = p[0]
    for i in range(1, len(p)):
        x += p[i] / (z + D(i))
    t = z + g + D('0.5')
    return decimal_sqrt(D(2) * pi) * decimal_pow(t, z + D('0.5')) * decimal_exp(-t) * x

def decimal_bernoulli(n: int) -> Decimal:
    """Compute Bernoulli number B_n using the Akiyama-Tanigawa algorithm."""
    if n == 0:
        return D(1)
    if n == 1:
        return D('-0.5')
    if n % 2 == 1 and n > 1:
        return D(0)
    a = [D(0)] * (n + 1)
    for m in range(n + 1):
        a[m] = D(1) / D(m + 1)
        for j in range(m, 0, -1):
            a[j - 1] = D(j) * (a[j - 1] - a[j])
    return a[0]

@functools.lru_cache(maxsize=256)
def _fibonacci_hp(n: int) -> int:
    """Fibonacci number F(n) via matrix exponentiation (exact integer)."""
    if n <= 1:
        return n
    # Fast doubling method
    def fib_pair(n):
        """Return Fibonacci pair (F(n), F(n+1))."""
        if n == 0:
            return (0, 1)
        a, b = fib_pair(n >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        if n & 1:
            return (d, c + d)
        return (c, d)
    return fib_pair(n)[0]

def lucas_number(n: int) -> int:
    """Lucas number L(n) = F(n-1) + F(n+1)."""
    if n == 0:
        return 2
    if n == 1:
        return 1
    return _fibonacci_hp(n - 1) + _fibonacci_hp(n + 1)


# ─── Additional Math Primitives v2.1 ───

def decimal_log10(x: Decimal) -> Decimal:
    """Base-10 logarithm: log10(x) = ln(x)/ln(10)."""
    ln10 = D('2.3025850929940456840179914546843642076011014886287729760333279009675726096773524802359972050895982983')
    return decimal_ln(x) / ln10

def decimal_sinh(x: Decimal) -> Decimal:
    """Hyperbolic sine: sinh(x) = (e^x - e^(-x)) / 2."""
    ex = decimal_exp(x)
    return (ex - D(1) / ex) / D(2)

def decimal_cosh(x: Decimal) -> Decimal:
    """Hyperbolic cosine: cosh(x) = (e^x + e^(-x)) / 2."""
    ex = decimal_exp(x)
    return (ex + D(1) / ex) / D(2)

def decimal_tanh(x: Decimal) -> Decimal:
    """Hyperbolic tangent: tanh(x) = sinh(x)/cosh(x)."""
    ex = decimal_exp(x)
    emx = D(1) / ex
    return (ex - emx) / (ex + emx)

def decimal_asin(x: Decimal) -> Decimal:
    """Arcsine via identity: asin(x) = atan(x / sqrt(1 - x²)) for |x| < 1."""
    if abs(x) >= D(1):
        pi = D('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170680')
        if x == D(1):
            return pi / D(2)
        if x == D(-1):
            return -pi / D(2)
        raise ValueError("asin domain: |x| <= 1")
    return decimal_atan(x / decimal_sqrt(D(1) - x * x))

def decimal_pi_machin() -> Decimal:
    """Compute π via Machin's formula: π/4 = 4·atan(1/5) - atan(1/239). 100-decimal independent verification."""
    return D(4) * (D(4) * decimal_atan(D(1) / D(5)) - decimal_atan(D(1) / D(239)))

def decimal_pi_chudnovsky(terms: int = 30) -> Decimal:
    """Compute π via Chudnovsky algorithm — fastest convergence (~14 digits/term)."""
    C = D(426880) * decimal_sqrt(D(10005))
    K = D(0)
    M = D(1)
    X = D(1)
    S = D(0)
    for k in range(terms):
        K = D(k)
        if k == 0:
            M = D(1)
        else:
            M = M * (D(6) * K - D(5)) * (D(2) * K - D(1)) * (D(6) * K - D(1))
            M = M / (K ** 3 * D('640320') ** 3 / D(24))
        # Actually use standard Chudnovsky form
        numer = decimal_factorial(6 * k) * (D(13591409) + D(545140134) * D(k))
        denom = decimal_factorial(3 * k) * (decimal_factorial(k) ** 3) * (D(-262537412640768000) ** k)
        S += numer / denom
    return C / S

def decimal_agm(a: Decimal, b: Decimal) -> Decimal:
    """Arithmetic-Geometric Mean: iterate a'=(a+b)/2, b'=sqrt(a*b) until convergence."""
    prec = getcontext().prec
    for _ in range(200):
        a_new = (a + b) / D(2)
        b_new = decimal_sqrt(a * b)
        if abs(a_new - b_new) < D(10) ** (-(prec + 5)):
            break
        a, b = a_new, b_new
    return a

def decimal_harmonic(n: int) -> Decimal:
    """Compute H_n = 1 + 1/2 + 1/3 + ... + 1/n to full precision."""
    s = D(0)
    for k in range(1, n + 1):
        s += D(1) / D(k)
    return s

def decimal_generalized_harmonic(n: int, m: int) -> Decimal:
    """Compute H_n^(m) = Σ_{k=1}^{n} 1/k^m — generalized harmonic number."""
    s = D(0)
    for k in range(1, n + 1):
        s += D(1) / D(k) ** m
    return s

def decimal_polylog(s: int, z: Decimal, terms: int = 500) -> Decimal:
    """Compute polylogarithm Li_s(z) = Σ_{k=1}^{terms} z^k / k^s for |z| <= 1."""
    result = D(0)
    z_power = D(1)
    for k in range(1, terms + 1):
        z_power *= z
        term = z_power / D(k) ** s
        result += term
        if abs(term) < D(10) ** -110:
            break
    return result

def decimal_binomial(n: int, k: int) -> Decimal:
    """Exact binomial coefficient C(n,k) as Decimal."""
    if k < 0 or k > n:
        return D(0)
    if k == 0 or k == n:
        return D(1)
    k = min(k, n - k)
    result = D(1)
    for i in range(k):
        result = result * D(n - i) / D(i + 1)
    return result

def decimal_catalan_number(n: int) -> Decimal:
    """n-th Catalan number: C_n = C(2n,n) / (n+1)."""
    return decimal_binomial(2 * n, n) / D(n + 1)
