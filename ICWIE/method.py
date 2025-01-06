from toolbox import *
from functools import reduce

labels =  [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11]

#Defining functions (photon counting, coincidence, and joint_probabiliy) that are required for the non-classically test of the ZWM experiment

def count_fun(psi, p):
    """
    Applies annihilation and creation operators to count photons in the given mode.
    
    Parameters:
        psi: The quantum state.
        p: The mode in which the photon count is performed.

    Returns:
        The quantum state after applying the photon counting operation.
    """
    photon_state = creation_fun(annihilation_fun(psi, p), p)
    return photon_state


def coincidence_fun(psi, p1, p2):
    """
    Simulates a coincidence detection event between two modes.

    Parameters:
        psi: The quantum state.
        p1: The first mode for coincidence detection.
        p2: The second mode for coincidence detection.

    Returns:
        The quantum state after applying the coincidence detection operation.
    """
    coincidence_state = creation_fun(
        creation_fun(
            annihilation_fun(
                annihilation_fun(psi, p1), p2
            ), p2
        ), p1
    )
    return coincidence_state


def joint_probability(psi, p1, p2):
    """
    Calculates the joint probability of a quantum state for two modes.

    Parameters:
        psi: The quantum state.
        p1: The first mode.
        p2: The second mode.

    Returns:
        The quantum state after applying the joint probability operation.
    """
    # Factorize the state by mode1[0] * mode2[m]
    dict_terms = sp.collect(psi, p1[0] * p2[m], evaluate=False)
    probability_terms = []

    # Process each term in the dual-rail representation
    for term in dict_terms.keys():
        if term != 1:  # Ignore constant terms
            # Replace symbolic terms with proper formatting
            formatted_term = term.replace(0, zero).replace(p1[n] * p2[m], p1[n, m])
            photon_indices = formatted_term.indices
            photon_count = photon_indices[1]  # Extract photon count

            if photon_count == zero:
                photon_count = photon_count.replace(zero, 0)
            formatted_term = formatted_term.replace(zero, 0)

            if photon_count != 0:
                expanded_term = sp.expand(dict_terms[p1[0] * p2[photon_count]] * p1[0] * p2[photon_count])
                probability_terms.append(expanded_term)
            else:
                probability_terms.append(0)

    return sum(probability_terms)

def encoded_label(nums, labels):
    # Map each index in nums to its corresponding label in labels
    encoded_labels = [labels[i] for i in nums]
    return encoded_labels

def sort(psi, labels):
    """
    Sorts terms in the quantum state `psi` by mapping indexed symbols to corresponding labels (sp.Wild)
    (e.g. a[0]*b[2] -> a[l1]*b[l2])
    
    Parameters:
        psi: A symbolic quantum state (SymPy expression).
        labels: A list of labels to map numeric indices to specific symbols (sp.Wild).

    Returns:
        A sorted expression with the indexed symbols replaced by their corresponding labels.
    """
    # Extract free symbols from the expression
    symbols = list(psi.free_symbols)
    
    # Identify indexed symbols and collect their bases
    bases = [sym.base for sym in symbols if isinstance(sym, sp.tensor.indexed.Indexed)]
    
    # Remove duplicates to identify unique bases
    unique_bases = list(set(bases))
    
    # Map numeric indices to labels
    mapped_labels = encoded_label(range(len(unique_bases)), labels)
    
    # Create a mapping between the unique bases and their corresponding labels
    mapping = list(zip(unique_bases, mapped_labels))
    
    # Construct the sorted expression 
    sorted_expression = [mapping[i][0][mapping[i][1]] for i in range(len(mapping))]
    
    sorted_result = reduce(lambda x, y: x * y, sorted_expression)
    
    return sorted_result


def rate_fun(psi, p1, p2, task, labels = labels):
    """
    Computes the rate of a quantum operation (coincidence, photon counting, or joint probability).

    Args:
        psi: The quantum state.
        p1: The first mode (e.g., photon mode).
        p2: The second mode (e.g., photon mode).
        task: The task to perform. Options are 'coincidence', 'photon_counting', or 'joint_probability'.

    Returns:
        The computed rate or 0 if the resulting state is null.
    """

    # Select the operation based on the task
    if task == 'coincidence':
        psi_transformed = coincidence_fun(psi, p1, p2)
    elif task == 'photon_counting':
        psi_transformed = count_fun(psi, p1)
    elif task == 'joint_probability':
        psi_transformed = joint_probability(psi, p1, p2)
    else:
        raise ValueError("Invalid task. Choose from 'coincidence', 'photon_counting', or 'joint_probability'.")

    # If the resulting state is null, return 0
    if psi_transformed == 0:
        return 0

    # Collect terms in the transformed and original states
    transformed_terms = sp.collect(psi_transformed, sort(psi_transformed, labels), evaluate=False)
    original_terms = sp.collect(psi, sort(psi, labels), evaluate=False)

    # Find the intersection of keys (common terms between the two states)
    common_terms = list(set(transformed_terms.keys()).intersection(original_terms.keys()))

    # Compute the rate by summing the coefficients of the common terms
    coefficients = []
    for term in common_terms:
        term_transformed = transformed_terms[term]
        term_original_conjugate = sp.conjugate(original_terms[term])
        coefficient = sp.expand(term_transformed * term_original_conjugate)
        coefficients.append(coefficient)

    # Sum up all coefficients to compute the rate
    rate = sum(coefficients)
    return rate

   
def visibility(r_min, r_max):
    vis = (r_max - r_min) / (r_max + r_min)
    return vis

def replace(psi, t):
    psi = sp.expand(psi.replace(sp.conjugate(sp.sqrt(1 - t**2)), sp.sqrt(1 - t**2)))
    return(psi)

def to_normalaze(psi, labels = labels):
    """
    Normalizes a given quantum expression by computing its normalization coefficient.

    Parameters:
        psi: The quantum expression to normalize.
        labels: A list of labels for sorting terms in the expression.

    Returns:
        The normalization coefficient for the expression.
    """
    if psi == zero  or psi == 0:
        return 0

    else:
        # Collect terms in the expression, grouped by sorted labels
        collected_terms =  sp.collect(psi, sort(psi, labels), evaluate = False)

        # Extract coefficients of each term
        TermsCoeff = list(collected_terms.items())
        CoefOfEachTerm = [(TermsCoeff[ii][1])**2 for ii in range(len(TermsCoeff))]

        # Compute the normalization coefficient
        NormalCoeff = 1 / sp.sqrt(sum(CoefOfEachTerm))

        return NormalCoeff 

def normalize_fun(psi, alpha, g):
    """
    Normalizes each term in a given expression based on the product of `alpha` and `g'.
    
    Parameters:
        psi: The expression to normalize.
        alpha: A symbol for grouping in the expression.
        g: A symbol  for grouping in the expression.

    Returns:
        The normalized and expanded final state.
    """
    # Collect terms grouped by alpha * g
    collected_terms = sp.collect(psi, alpha * g, evaluate=False)
    
    # Normalize each term
    terms = list(collected_terms.values())
    for idx, state in enumerate(terms):
        normalized_state = sp.expand(to_normalaze(state) * state)
        terms[idx] = normalized_state
    
    # Map the normalized terms back to their keys
    normalized_dict = dict(zip(collected_terms.keys(), terms))
    items = list(normalized_dict.items())
    
    # Reconstruct the final state
    final_state = sum(item[0] * item[1] for item in items)
    final_state = sp.expand(final_state)
    
    return final_state
