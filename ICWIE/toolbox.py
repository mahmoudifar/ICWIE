import sympy as sp

num_mode = 3 
zero = sp.Symbol('zero')
coeff = 1 / sp.sqrt(2) 
path = ['a', 'b', 'c'] + [f'{prefix}{i}' for i in range(1, num_mode + 1) for prefix in ['p', 's', 'i']]

for idx, op in enumerate(path): #op: optical path
    globals()[op] = sp.IndexedBase(op)

n, m, l, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11 = map(sp.Wild, 
        ['n', 'm', 'l', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'l11']
                                                            )

alpha, beta, phi, phi_i, phi_s, phi_r, phi_t, phi_0, t,\
        t1, t2, r, r1, r2, g, h = sp.symbols('alpha beta phi phi_i phi_s phi_r\
                                       phi_t phi_0 t t1 t2 r r1 r2 g h', integer = True
                                       )

# Define the photon annihilation operator
def annihilation_fun(psi, a, state='fock'):
    """
    Applies the annihilation operator to the given state.

    Parameters:
        psi: sympy expression
            The state representation.
        a: spmpy.IndexedBase
            The path of the photon.
        state: str
            The type of state is either 'fock' or 'coherent state'.

    Returns:
        sympy expression: The updated state after applying the annihilation operator.
        
    Raises:
        ValueError: If the `state` parameter is not a recognized type.
    """
    
    if state == 'fock':
        psi = psi.replace(a[n], sp.sqrt(n) * a[n - 1])
    elif state == 'coherent':
        psi = psi.replace(a[n], n * a[n])  # Use n as a complex number
    else:
        raise ValueError(f"Invalid state type: '{state}'. Expected 'fock' or 'coherent'.")
        
    return psi

# Define the photon creation operator
def creation_fun(psi, a, state='fock', n_cutoff=None):
    """
    Applies the creation operator to a quantum state.

    Parameters:
        psi: sympy expression
            The symbolic quantum state representation.
        a: sympy.IndexedBase
            The path of the photon.
        state: str
            The type of quantum state is either 'fock' or 'coherent'. Defaults to 'fock'.
        n_cutoff: int, optional
            The truncation point for coherent state expansion. Required for the 'coherent' state.

    Returns:
        sympy expression: The updated state after applying the creation operator.

    Raises:
        ValueError: If `state` is invalid or `n_cutoff` is missing for a 'coherent' state.
    """
    
    def creation_operator(psi, a):
        """
        Applies the creation operator on the Fock state basis.
        """
        return psi.replace(a[n], sp.sqrt(n + 1) * a[n + 1])
        
    if state == 'fock':
        # Apply creation operator in Fock state basis
        psi = creation_operator(psi, a)
        
    elif state == 'coherent':
         if n_cutoff is None:
            raise ValueError("Parameter `n_cutoff` is required for a coherent state.")
          # Construct the coherent state based on the fock state and apply the creation operator
         psi_expan = coherent_state(psi, a, n_cutoff)
         psi = creation_operator(psi_expan, a)
    else:
        raise ValueError(f"Invalid state type: '{state}'. Expected 'fock' or 'coherent'.")

    return psi

def coherent_state(psi, a, n_cutoff):
    """
    Calculate the coherent state.

    Parameters:
    psi: sympy expression
        The symbolic quantum state representation.
    n: sympy.Symbol 
        The coherent state parameter.
    a: sympy.IndexedBase
        The indexed base for the number (Fock) states.
    n_cutoff : int
        The maximum value of n for summation.

    Returns:
    sympy expression
        The coherent state is a symbolic expression.
    """
    psi = psi.replace(a[n], sp.exp(-abs(n ** 2) / 2) *
                      sp.summation((n ** m / sp.sqrt(sp.factorial(m))) * a[m],
                                   (m, 0, n_cutoff)
                                  )
                     )         
    
    return sp.expand(psi)  # Expand the result for clarity


# Define a function for each optical device (beam splitter, spdc, phase shifter, and tritter) applied to our experimental setup.

def beamsplitter(psi, p1, p2, r=1/sp.sqrt(2), phi_r=0, phi_t=0, phi_0=0):
    """
    Simulates the action of a beamsplitter on a quantum state.

    Parameters:
        psi: sympy expression
            The input quantum state.
        p1, p2: sympy.IndexedBase
            Mode creation operators.
        r: float or sympy symbol, optional
            Reflectivity of the beamsplitter (default is 1/sqrt(2)).
        phi_r: float or sympy symbol, optional
            Phase shift for reflection (default is 0).
        phi_t: float or sympy symbol, optional
            Phase shift for transmission (default is 0).
        phi_0: float or sympy symbol, optional
            Global phase (default is 0).

    Returns:
        sympy expression:
            The transformed quantum state.
    """
    
    global a, b
    # Compute transmission coefficient
    t = sp.sqrt(1 - r ** 2)
    
    # Define the beamsplitter transformations
    trans0 = a * r * sp.exp(sp.I * (-phi_r + phi_0)) + b * t * sp.exp(sp.I * (phi_t + phi_0))
    trans1 = a * t * sp.exp(sp.I * (-phi_t + phi_0)) - b * r * sp.exp(sp.I * (phi_r + phi_0))

    # Decompose the input expression into a dual-rail representation
    dictated = sp.collect(psi, [p1[n] * p2[m]], evaluate=False)
    dual_rail_representation = list(dictated.keys())
    
    for idx, term in enumerate(dual_rail_representation):
        if term != 1:
            # Replace for photon count
            state = term.replace(0, zero)
            state1 = state.replace(p1[n] * p2[m], p1[n, m])
            num_photon = state1.indices  # Extract photon number information
            N, M = num_photon
            
            # Convert 'zero' back to 0 for numeric comparison
            N = 0 if N == zero else N
            M = 0 if M == zero else M
            
            # Replace with normalized state representation
            state = state.replace(p1[n] * p2[m], p1[0] * p2[0] / sp.sqrt(sp.factorial(N) * sp.factorial(M)))
            state = state.replace(zero, 0)
            
            if N == 0 and M == 0:
                pass # Do nothing for vacuum states
            elif N == 0: # M > 0
                for _ in range(M):
                    state = sp.expand(trans1.xreplace({a: creation_fun(state, p1), b: creation_fun(state, p2)}))
            elif M == 0: # N > 0
                for _ in range(N):
                    state = sp.expand(trans0.xreplace({a: creation_fun(state, p1), b: creation_fun(state, p2)}))
            else:  # Both N and M > 0
                for _ in range(N):
                    state = sp.expand(trans0.xreplace({a: creation_fun(state, p1), b: creation_fun(state, p2)}))
                for _ in range(M):
                    state = sp.expand(trans1.xreplace({a: creation_fun(state, p1), b: creation_fun(state, p2)}))
            
            dual_rail_representation[idx] = state

    # Construct the output state
    psi_out = dict(zip(dual_rail_representation, list(dictated.values())))
    transformed_state = sp.expand(sp.Add(*[mode1 * mode2 for mode1, mode2 in psi_out.items()]))
    
    return transformed_state

def phase_shifter(psi, p, phi, state='fock'):
    """
    Applies a phase shift to a quantum state.

    Parameters:
        psi: sympy expression
            The input quantum state.
        p: sympy.IndexedBase
            The photon path or mode.
        phi: float or sympy symbol
            The phase shift to apply.
        state: str, optional
            The type of quantum state representation ('fock' or 'coherent').  The default is 'fock'.

    Returns:
        sympy expression
            The quantum state with the phase shift applied.
    """
    phase_transform = {
        'fock': lambda n: sp.exp(sp.I * n * phi) * p[n],
        'coherent': lambda n: p[n * sp.exp(sp.I * phi)]
    }
    
    if state in phase_transform:
        psi = psi.replace(p[n], phase_transform[state](n))
    else:
        raise ValueError(f"Unsupported state type: '{state}'. Supported types are 'fock' and 'coherent'.")
    
    return psi
    
# Simulating a tritter (three-port optical device) using two beam splitters
def tritterA(psi, p1, p2, p3, r1=1/sp.sqrt(3), r2=1/sp.sqrt(2)):
    psi = beamsplitter(beamsplitter(psi, p1, p2, r1), p2, p3, r2)
    return psi

def tritterB(psi, p1, p2, p3, phi_s=sp.pi):
    psi = phase_shifter(tritterA(psi, p1, p2, p3), p2, phi_s)
    return psi

def spdc(psi, p1, s1, i1, state='fock'):
    """
    Simulate a Spontaneous Parametric Down-Conversion (SPDC) process.

    Parameters:
    psi: sympy expression 
        The initial state.
    p1, s1, i1: sympy.IndexedBase
        Lists of paths for pump, signal, and idler modes respectively.
    state: str
        The state type is either 'fock' or 'coherent'.

    Returns:
    sympy expression: 
        The final state after SPDC process.
    """
    m, l = 0, 0  # Assuming these are the number of photons for the s1, i1 for the initial state

    # Replace the initial state |psi>_p  with the |psi>_p|0>_s|0>_i
    psi = psi.replace(p1[n], p1[n] * s1[m] * i1[l])
    
    collected_terms = sp.collect(psi, [p1[n] * s1[m] * i1[l]], evaluate=False)
    terms = list(collected_terms.keys())

    for idx, term in enumerate(terms):
        if term != 1:
            if state == 'coherent':
                expr1 = g * annihilation_fun(creation_fun(creation_fun(term, s1), i1), p1, state='coherent')
                expr2 = (g ** 2 * annihilation_fun(
                    annihilation_fun(creation_fun(creation_fun(creation_fun(creation_fun(term, s1), s1), i1), i1), p1, state='coherent'),
                    p1, state='coherent')) / 2
                output_state = term + expr1 + expr2
            elif state == 'fock':
                expr1 = g * annihilation_fun(creation_fun(creation_fun(term, s1), i1), p1)
                expr2 = (g ** 2 * annihilation_fun(
                    annihilation_fun(creation_fun(creation_fun(creation_fun(creation_fun(term, s1), s1), i1), i1), p1),
                    p1)) / 2
                output_state = term + expr1 + expr2
            else:
                raise ValueError(f"Unsupported state type: '{state}'. Supported types are 'fock' and 'coherent'.")
            
            terms[idx] = output_state

    # Reconstruct the dictionary with new terms
    psi_out = dict(zip(terms, list(collected_terms.values())))
    
    # Combine terms into a single expression
    transformed_state = sp.expand(sp.Add(*[mode1 * mode2 for mode1, mode2 in psi_out.items()]))
    
    return transformed_state

#The indistinguishability between idler photons emerges from the perfect alignment of their paths (path identity).

def pathidentity(psi, i1, i2, phi_i=0):
    """
    Processes a symbolic quantum state to apply path identity transformations.
    i1[n] * i2[m] -> sqrt(binomial(n + m, n)) * i1[0] * i2[n + m] * exp(I * m * phi_i)

    Parameters:
        psi: sympy expression
            Symbolic expression representing the quantum state.
        i1, i2: sympy.IndexedBase
            Path of idler photons.
        phi_i: float or sympy symbol, optional
            The phase change is due to propagation from crystal NL1 to crystal NL2 (default is 0).

    Returns:
        sympy expression: 
            A symbolic expression of the final state after transformations.
    """

    # Collect terms grouped by products of i1[n] and i2[m]
    collected_terms = sp.collect(psi, [i1[n] * i2[m]], evaluate=False)

    updated_terms = []
    # Iterate through collected terms and apply the transformation
    for term1, term2 in collected_terms.items():
        # Replace the pattern i1[n] * i2[m] with the modified expression
        modified_term1 = term1.replace(0, zero).replace(
            i1[n] * i2[m], sp.sqrt(sp.binomial(n + m, n)) * i1[0] * i2[n + m] * sp.exp(sp.I * m * phi_i)
        )
        updated_terms.append(modified_term1.replace(zero, 0) * term2)
        
    # Construct the output state
    transformed_state = sp.expand(sum(updated_terms))

    return transformed_state
