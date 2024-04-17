import sympy as sp

num_mode = 3 
zero = sp.Symbol('zero')
coeff = 1/sp.sqrt(2) 
path = ['a', 'b', 'c'] + [f'{prefix}{i}' for i in range(1, num_mode + 1) for prefix in ['p', 's', 'i']]

for idx, op in enumerate(path): #op: optical path
    globals()[op] = sp.IndexedBase(op)

n, m, l, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11 = map(sp.Wild, ['n', 'm', 'l', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'l11'])
alpha, phi_s, g, h, r, t, t2, t2, r1, r2 = sp.symbols('alpha, phi_s g h r t t1 t2 r1 r2', integer = True)

# Define the photon annihilation operator
def annihilation_fun(psi, a, state = 'fock'):
    if state == 'fock':
        psi = psi.replace(a[n], sp.sqrt(n) * a[n - 1])
    elif state == 'coherent state':
        psi = psi.replace(a[n], n * a[n])  # Use n as a complex number
    return psi

# Define the photon creation operator
def creation_fun(psi, a):
    psi = psi.replace(a[n], sp.sqrt(n + 1) * a[n + 1])
    return psi

#define a function for each optical device (Beam splitter, SPDC, Phase shifter, Tritter) that are applied in our experimental setup.

def beamsplitter(expr, p1, p2, r = coeff):
    global a, b
    t = sp.sqrt(1 - r**2)
    expr0 = a*r + b*t # a and b is a creation operator.
    expr1 = b*r - a*t 
    dictadd = sp.collect(expr, [p1[n] * p2[m]], evaluate = False)
    dual_rail_representation = list(dictadd.keys())
    for idx, term in enumerate(dual_rail_representation):
        if term != 1 :
            item = term.replace(0, zero)
            item1 = item.replace(p1[n]*p2[m], p1[n, m])
            num_photon = item1.indices
            N = num_photon[0]
            M = num_photon[1]
            if N == zero:
                N = N.replace(zero, 0)
            if M == zero:
                M = M.replace(zero, 0)    
            item = item.replace(p1[n]*p2[m], p1[0]*p2[0]/sp.sqrt(sp.factorial(n)*sp.factorial(m)))
            item = item.replace(zero, 0)
            if N == 0 and M == 0:
                item = item
            elif N == 0 :
                for _ in range(M):
                    item = sp.expand(expr1.xreplace({a: creation_fun(item, p1), b: creation_fun(item, p2)}))
            elif M == 0 :
                for _ in range(N):
                    item = sp.expand(expr0.xreplace({a: creation_fun(item, p1), b: creation_fun(item, p2)}))
            elif N > 0 and M > 0: 
                for _ in range(N):
                    item = sp.expand(expr0.xreplace({a: creation_fun(item, p1), b: creation_fun(item, p2)}))
                for _ in range(M):
                    item = sp.expand(expr1.xreplace({a: creation_fun(item, p1), b: creation_fun(item, p2)}))  
            dual_rail_representation[idx] = item 
                
    d = dict(zip(dual_rail_representation, list(dictadd.values()))) 
    get_to_items = list(d.items())
    PHI = [get_to_items[i][0]*get_to_items[i][1] for i in range(len(get_to_items ))]
    phi = sum(PHI)
    final_state = sp.expand(phi)    
    return final_state

def phaseshifter(psi, a, phi_s):
    psi = psi.replace(a[n], sp.exp(sp.I*n*phi_s)*a[n])
    return psi

def tritterA(psi, p1, p2, p3, r1 = 1/sp.sqrt(3), r2 = 1/sp.sqrt(2)):
    psi = beamsplitter(beamsplitter(psi, p1, p2, r1), p2, p3, r2)
    return(psi)

def tritterB(psi, p1, p2, p3, phi_s = sp.pi):
    psi = phaseshifter(tritterA(psi, p1, p2, p3), p2, phi_s)
    return(psi)

def spdc(psi, p1, s1, i1, state = 'fock'):
    psi = psi.replace(p1[n], p1[n]*s1[0]*i1[0])
    dictadd = sp.collect(psi, [p1[n]*s1[m]*i1[l]], evaluate = False)
    term = list(dictadd.keys())  
    for idx, item in enumerate(term):
        if item != 1 :
            if state == 'coherent state':
                expr1 = g*annihilation_fun(creation_fun(creation_fun(item, s1), i1), p1, state ='coherent state')
                expr2 = g**2*annihilation_fun(annihilation_fun(creation_fun(creation_fun(creation_fun(creation_fun(item, s1),s1), i1), i1), p1, state ='coherent state'), p1, state ='coherent state')/2
                output_state = item + expr1 + expr2 
                term[idx] = output_state
            elif state == 'fock':
                expr1 = g* annihilation_fun(creation_fun(creation_fun(item, s1), i1), p1)
                expr2 = g**2*annihilation_fun(annihilation_fun(creation_fun(creation_fun(creation_fun(creation_fun(item, s1),s1), i1), i1), p1), p1)/2
                #expr3 = h**2*creation_fun(annihilation_fun(annihilation_fun(creation_fun(annihilation_fun(creation_fun(item, s1),s1),i1), i1), p1),p1)
                output_state = item + expr1 + expr2 #+expr3
                term[idx] = output_state
            else:
                raise ValueError( f"introduce valid state.")        
    d = dict(zip(term, list(dictadd.values())))
    get_to_items = list(d.items())
    PHI = [get_to_items[i][0]*get_to_items[i][1] for i in range(len(get_to_items))]
    final_state = sum(PHI)
    final_state = sp.expand(final_state)
    return(final_state) 

#The indistinguishability between idler photons emerges from the perfect alignment of their paths (path identity).

def pathidentity(psi, i1, i2):
    dictadd = sp.collect(psi, [i1[n]*i2[m]], evaluate = False)
    expr = list(dictadd.keys())
    for idx, item in enumerate(expr):
        item1 = item.replace(0, zero)
        item1= item1.replace(i1[n]*i2[m], i1[n, m])
        num_photon = item1.indices
        N = num_photon[0]
        M = num_photon[1]
        if N == zero:
            N = N.replace(zero, 0)
        if M == zero:
            M = M.replace(zero, 0)
        K = M + N
        PHI = i1[0]*i2[0]
        PSI = i1[0]*i2[0]
        if N == 0 and M == 0:
            PHI = PHI
        else:
            for _ in range(K):
                PHI = creation_fun(PHI, i2)
            for _ in range(M):
                PSI = creation_fun(PSI, i2)
            for _ in range(N):
                PSI = creation_fun(PSI, i1)
            PSI = PSI/item
            PHI = PHI/PSI
        expr[idx] = PHI
    get_to_items = list(zip(expr, list(dictadd.values()))) 
    phi0 = [get_to_items[i][0]*get_to_items[i][1] for i in range(len(get_to_items))]
    phi = sum(phi0)
    final_state = sp.expand(phi)    
    return final_state   