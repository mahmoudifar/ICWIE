from toolbox import *
from functools import reduce

num =  [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11]

#Defining functions (photon counting, coincidence, and joint_probabiliy) that are required for the non-classically test of the ZWM experiment

def photon_fun(psi, p1):
    photon = creation_fun(annihilation_fun(psi, p1), p1)
    return photon

def coincidence_fun(psi, p1, p2):
    coincidence = creation_fun(creation_fun(annihilation_fun(annihilation_fun(psi, p1), p2), p2), p1)
    return coincidence

def joint_probability(psi, p1 ,p2):
    dictaad = sp.collect(psi, p1[0]*p2[m], evaluate = False)
    psi2 = []
    dual_rail_representation = list(dictaad.keys())
    for term in dual_rail_representation:
        if term != 1 :
            item = term.replace(0, zero)
            item1 = item.replace(p1[n]*p2[m], p1[n,m])
            num_photon = item1.indices
            M = num_photon[1]
            if M == zero:
                 M = M.replace(zero, 0)    
            item = item.replace(zero, 0)
            if M != 0:
                psi1 = sp.expand(dictaad[p1[0]*p2[M]]*p1[0]*p2[M])
                psi2.append(psi1)
            else:
                psi1 = 0
                psi2.append(psi1)
    return sum(psi2)

def encoded_label(nums, labels):# for transform num to alphabet
    encodedlabels = [labels[ii] for ii in nums]
    return encodedlabels

def sort (psi, numm):
    expr = list(psi.free_symbols) 
    base = []
    for ii in expr:
        if type(ii)==sp.tensor.indexed.Indexed:
             base.append(ii.base)
    path = list(set(base))
    num1 = [i for i in (range(len(path)))]
    num1 = encoded_label(num1, numm)
    phi = list(zip(path, num1))
    PHI = [phi[i][0][phi[i][1]] for i in range(len(phi))]
    expr1 = reduce(lambda x, y: x*y, PHI)
    return(expr1) 

def rate_fun(psi, p1, p2, task):
    global num
    if task == 'coincidence':
        psi0 = coincidence_fun(psi, p1, p2)
    elif task =='photon_counting':
        psi0 = photon_fun(psi, p1)
    elif task =='joint_probability':
        psi0 = joint_probability(psi, p1 ,p2)
    else:
        raise ValueError( f"introduce valid task.")   
    
    if psi0 == 0: return 0
    else:
        dictted1 = sp.collect(psi0, sort(psi0, num), evaluate = False)
        dictted2 = sp.collect(psi, sort(psi, num), evaluate = False)
        intersection = list(set(dictted1.keys()).intersection(dictted2.keys()))
        coefficient= []
        for jj in intersection:
            term1 = dictted1[jj]
            term2 =sp.conjugate(dictted2[jj])
            term = sp.expand((term1*term2))
            coefficient.append(term)
        rate = sum(coefficient)
        return rate
    
def visibility(r_min, r_max):
    vis =(r_max - r_min)/((r_max + r_min))
    return vis

def replace(psi, t):
    psi = sp.expand(psi.replace(sp.conjugate(sp.sqrt(1 - t**2)), sp.sqrt(1 - t**2)))
    return(psi)

def to_normalaze(expr):
    global num
    if expr == zero:return 0
    else:
        dictadd =  sp.collect(expr, sort (expr, num), evaluate = False)
        TermsCoeff = list(dictadd.items())
        CoefOfEachTerm = [(TermsCoeff[ii][1])**2 for ii in range(len(TermsCoeff))]
        NormalCoeff = 1/sp.sqrt(sum(CoefOfEachTerm))
        return NormalCoeff 
    
def normalaze_fun(expr, alpha, g):
    dictadd = sp.collect(expr, alpha*g , evaluate = False)
    term = list(dictadd.values())
    for idx, state in enumerate(term):
        item = sp.expand(to_normalaze(state)*state)
        term[idx] = item  
    d = dict(zip(list(dictadd.keys()), state))
    get_to_items = list(d.items())
    PHI = [get_to_items[i][0]*get_to_items[i][1] for i in range(len(get_to_items))]
    final_state = sum(PHI)
    final_state = sp.expand(final_state)
    return(final_state)
