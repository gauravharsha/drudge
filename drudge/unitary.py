"""
Drudge for reduced AGP - PBCS Hamiltonian.
"""
import collections, functools, operator

from sympy import Integer, Symbol, IndexedBase, KroneckerDelta, factorial
from sympy.utilities.iterables import default_sort_key

from drudge import Tensor
from drudge import PartHoleDrudge, SpinOneHalfPartHoleDrudge
from agp import *
from drudge.canon import IDENT,NEG
from drudge.canonpy import Perm
from drudge.term import Vec, Range, Term
from drudge.utils import sympy_key
from drudge.genquad import GenQuadDrudge


class UnitaryGroupDrudge(GenQuadDrudge):
    r"""Drudge to manipulate and work with Unitary Group Generators defined as:
    ..math::

        E_{p,q} = c_{p,\uparrow}^\dagger c_{q,\uparrow} + \mathrm{h.c.}

    where the symbols :math: `p` and :math: `q` denote a general orbital index.
    This Unitary Group drudge comtains utilites to work with either the pairing
    Hamiltonian or any other system in the language of the unitary group gens.

    Without the presence of a reference, such as the Fermi sea or the pairing-
    ground state, there is no one way to define a canonical ordering for a string
    of :math: `E_{p,q}'\mathrm{s}`.

    Here we follow an lexicological or alphabetical ordering based on the first
    index followed by the second. For example, following is a representative of 
    the correct canonical ordering:
    ..math::
        E_{p,q} E_{p,r} E_{q,r} E_{r,p}

    """

    DEFAULT_GEN = Vec('E')

    def __init__(
            self, ctx,
            # part_range=Range('V', 0, Symbol('nv')),
            # part_dumms=PartHoleDrudge.DEFAULT_PART_DUMMS,
            # hole_range=Range('O', 0, Symbol('no')),
            # hole_dumms=PartHoleDrudge.DEFAULT_HOLE_DUMMS,
            all_orb_range=Range('A' ,0, Symbol('na')),
            all_orb_dumms=PartHoleDrudge.DEFAULT_ORB_DUMMS,
            energies=IndexedBase(r'\epsilon'), interact=IndexedBase('G'),
            ug_gen=DEFAULT_GEN,
            **kwargs
    ):
        """Initialize the drudge object."""

        # Initialize the base su2 problem.
        super().__init__(ctx, **kwargs)

        # Set the range and dummies.
        # self.part_range = part_range
        # self.hole_range = hole_range
        # self.set_dumms(part_range, part_dumms)
        # self.set_dumms(hole_range, hole_dumms)
        # self.add_resolver_for_dumms()

        self.all_orb_range = all_orb_range
        self.all_orb_dumms = tuple(all_orb_dumms)
        self.set_dumms(all_orb_range, all_orb_dumms)
        self.set_name(*self.all_orb_dumms)
        self.add_resolver({
            i: (all_orb_range) for i in all_orb_dumms
        })

        # Set the operator attributes

        self.ug_gen = ug_gen

        # Make additional name definition for the operators.
        self.set_name(**{
            ug_gen.label[0]+'_':ug_gen,
        })

        self.set_name(ug_gen)
        
        # Defining spec for passing to an external function - the Swapper
        spec = _UGSpec(
                ug_gen=ug_gen,
        )

        self._spec = spec
        
        # Create an instance of the ProjectedBCS or AGP Drudge to map from E_pq
        # to D_dag, N, D so that normal ordering and vev evaluation can be done
        agp_dr = ProjectedBCSDrudge(
            ctx, all_orb_range,all_orb_dumms
        )
        self._agp_dr = agp_dr

        # Mapping to D_dag, N, D
        D_p = agp_dr.raise_
        N_ = agp_dr.cartan
        D_m = agp_dr.lower
        sig = agp_dr.sig
        eta = agp_dr.eta
        self.eta = eta
        self.sig = sig

        gen_idx1, gen_idx2 = self.all_orb_dumms[:2]

        epq_def = self.define(
            ug_gen,gen_idx1,gen_idx2,
            sig[gen_idx1,gen_idx2]*( eta[gen_idx1]*D_m[gen_idx1,gen_idx2] + \
                eta[gen_idx2]*D_p[gen_idx1,gen_idx2] ) + \
                KroneckerDelta(gen_idx1,gen_idx2)*N_[gen_idx1]
        )

        self._defs = [
            epq_def
        ]

        # set the Swapper
        self._swapper = functools.partial(_swap_ug, spec=spec)

    @property
    def swapper(self) -> GenQuadDrudge.Swapper:
        """Swapper for the new AGP Algebra."""
        return self._swapper

    def _transl2agp(self, tensor: Tensor):
        """Translate a tensor object in terms of the fermion operators.

        This is an internally utility.  The resulted tensor has the internal
        fermion drudge object as its owner.
        """
        return Tensor(
            self._agp_dr,
            tensor.subst_all(self._defs).terms
        )
    def get_vev_agp(self,h_tsr: Tensor, Ntot: Symbol):
        """Function to evaluate the expectation value of a normal
        ordered tensor 'h_tsr' with respect to the projected BCS
        or the AGP state. This can be done after translating the the
        unitary group terms into the AGP basis algebra.
            h_tsr = tensor whose VEV is to be evaluated
            Ntot = total number of orbitals available
        """
        transled = self._transl2agp(h_tsr)
        transled = self._agp_dr.agp_simplify(transled,final_step=True)
        res = self._agp_dr.get_vev(transled,Ntot)
        return Tensor(self, res.terms)


_UGSpec = collections.namedtuple('_UGSpec',[
    'ug_gen'
])


def _swap_ug(vec1: Vec, vec2: Vec, depth=None, *,spec: _UGSpec):
    """Swap two vectors based on the commutation rules for Unitary Group generators.
    Here, we introduce an additional input parameter 'depth' which is never
    specified by the user. Rather, it is put to make use os the anti-symmetric 
    nature of the commutation relations and make the function def compact. 
    """
    if depth is None:
        depth = 1
    
    char1, indice1, key1 = _parse_vec(vec1,spec)
    char2, indice2, key2 = _parse_vec(vec2,spec)
    
    p = indice1[0]
    q = indice1[1]
    r = indice2[0]
    s = indice2[1]

    if p != r:
        if p is min(p,r,key=default_sort_key):
            return None
        elif r is min(p,r,key=default_sort_key):
            expr = KroneckerDelta(q,r)*spec.ug_gen[p,s] - KroneckerDelta(p,s)*spec.ug_gen[r,q]
            return _UNITY, expr
        else:
            return None
    elif p == r:
        if q is min(q,s,key=default_sort_key):
            return None
        elif s is min(q,s,key=default_sort_key):
            expr = KroneckerDelta(q,r)*spec.ug_gen[p,s] - KroneckerDelta(p,s)*spec.ug_gen[r,q]
            return _UNITY, expr
        else:
            return None
    else:
        assert False



_UG_GEN = 0

_UNITY = Integer(1)
_NOUGHT = Integer(0)
_TWO = Integer(2)

def _parse_vec(vec, spec: _UGSpec):
    """Get the character, lattice indices, and the indices of keys of vector.
    """
    base = vec.base
    if base == spec.ug_gen:
        char = _UG_GEN
    else:
        raise ValueError('Unexpected generator of the Unitary Group',vec)
    
    indices = vec.indices
    if len(indices)!=2:
        raise ValueError('Unexpected length of indices for the generators of Unitary Group',vec)
    keys = tuple(sympy_key(i) for i in indices)

    return char, indices, keys
