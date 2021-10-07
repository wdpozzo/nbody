cimport cython
from libc.math cimport abs, sqrt
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _create_body(body_t *b,
                       long double mass,
                       long double x,
                       long double y,
                       long double z,
                       long double px,
                       long double py,
                       long double pz,
                       long double sx,
                       long double sy,
                       long double sz) nogil:
    
    b.mass = mass
    b.q[0]    = x
    b.q[1]    = y
    b.q[2]    = z
    b.p[0]    = px
    b.p[1]    = py
    b.p[2]    = pz
    b.s[0]    = sx
    b.s[1]    = sy
    b.s[2]    = sz

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _create_system(body_t *b,
                         unsigned int n,
                         long double[:] mass,
                         long double[:] x,
                         long double[:] y,
                         long double[:] z,
                         long double[:] px,
                         long double[:] py,
                         long double[:] pz,
                         long double[:] sx,
                         long double[:] sy,
                         long double[:] sz) nogil:

    cdef unsigned int i

    for i in range(n):
        _create_body(&b[i], mass[i], x[i], y[i], z[i], px[i], py[i], pz[i], sx[i], sy[i], sz[i])

    return
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _merge_bodies(body_t *b, unsigned int i_survive, unsigned int i_remove) nogil:
    return

cdef void _fits_setup(fit_coefficients_t *out, long double m1, long double m2, long double chi1, long double chi2) nogil:
    """
    Computes auxiliary variables for UIB final-state and luminosity fit functions
    
    :param m1: primary mass in units of solar masses, :math:`m_1\\geq m_2`
    :type m1: float
    :param m2: secondary mass in units of solar masses
    :type m2: float
    :param chi1: dimensionless spin of the primary mass, :math:`z` component
    :type chi1: float
    :param chi2: dimensionless spin of the secondary mass, :math:`z` component
    :type chi2: float
    
    :return:
     - total mass :math:`M` in units of solar masses
     - symmetric mass ratio :math:`\\eta = m_1\\cdot m_2/M^2`
     - :math:`\\eta^2`
     - :math:`\\eta^3`
     - :math:`\\eta^4`
     - total aligned spin :math:`S_{\\rm tot} = (\\chi_1\\cdot m_1^2+\\chi_2 \\cdot m_2^2)/M^2`
     - effective aligned spin :math:`\\hat{S} = S_{\\rm tot}\\cdot M^2/(m_1^2+m_2^2)`
     - :math:`\\hat{S}^2`
     - :math:`\\hat{S}^3`
     - :math:`\\hat{S}^4`
     - spin difference :math:`\\chi_{\\rm diff} = \\chi_1 - \\chi_2`
     - :math:`\\chi_{\\rm diff}^2`
     - :math:`\\sqrt{2}`
     - :math:`\\sqrt{3}`
     - :math:`\\sqrt{(1 - 4\\eta)}`
    :rtype: tuple
    """
    cdef long double tmp_m, tmp_spin
    
    # fit assumes m1 > m2, so in case swap the bodies
    if m2 > m1:
        tmp_m = m2
        m2 = m1
        m1 = tmp_m
        tmp_spin = chi2
        chi2 = chi1
        chi1 = tmp_spin

    # binary masses
    out.m    = m1+m2
    if out.m <=0 :
      raise ValueError("m1+m2 must be positive")
    cdef long double msq  = out.m*out.m
    cdef long double m1sq = m1*m1
    cdef long double m2sq = m2*m2

    # symmetric mass ratio
    out.eta  = m1*m2/msq
    out.eta2 = out.eta*out.eta
    out.eta3 = out.eta2*out.eta
    out.eta4 = out.eta2*out.eta2

    # spin variables (in m = 1 units)
    cdef long double S1    = chi1*m1sq/msq # spin angular momentum 1
    cdef long double S2    = chi2*m2sq/msq # spin angular momentum 2
    out.Stot  = S1+S2         # total spin
    out.Shat  = (chi1*m1sq+chi2*m2sq)/(m1sq+m2sq) # effective spin, = msq*Stot/(m1sq+m2sq)
    out.Shat2 = out.Shat*out.Shat
    out.Shat3 = out.Shat2*out.Shat
    out.Shat4 = out.Shat2*out.Shat2

    # spin difference, assuming m1>m2
    out.chidiff  = chi1 - chi2
    out.chidiff2 = out.chidiff*out.chidiff

    # typical squareroots and functions of eta
    out.sqrt2      = sqrt(2.)
    out.sqrt3      = sqrt(3.)
    out.sqrt1m4eta = (1. - 4.*out.eta)**0.5

    return

def final_mass(long double m1, long double m2, long double chi1, long double chi2, unsigned int version = 2):
    return _final_mass(m1, m2, chi1, chi2, version)

cdef double _final_mass(long double m1, long double m2, long double chi1, long double chi2, unsigned int version) nogil:
    """
    | Calculates the final mass with the aligned-spin NR fit
     by `Xisco Jimenez Forteza, David Keitel, Sascha Husa et al. <https://arxiv.org/abs/1611.00332>`_
    | Versions :math:`v_1` and :math:`v_2` use the same ansatz, with :math:`v_2` calibrated to additional SXS and RIT data

    :param m1: primary mass in units of solar masses, :math:`m_1\\geq m_2`
    :type m1: float
    :param m2: secondary mass in units of solar masses
    :type m2: float
    :param chi1: dimensionless spin of the primary mass, :math:`z` component
    :type chi1: float
    :param chi2: dimensionless spin of the secondary mass, :math:`z` component
    :type chi2: float
    :param version: fit version, either :math:`v_1` or :math:`v_2`. *Optional*, default is :math:`v_2`
    :type version: string
    
    :return: final mass :math:`M_f` in units of solar masses
    :rtype: float
    """
    
    cdef long double a2
    cdef long double a3
    cdef long double a4
    cdef long double b1
    cdef long double b2
    cdef long double b3
    cdef long double b5
    cdef long double f20
    cdef long double f30
    cdef long double f50
    cdef long double f10
    cdef long double f21
    cdef long double d10
    cdef long double d11
    cdef long double d20
    cdef long double d30
    cdef long double d31
    cdef long double f11
    cdef long double f31
    cdef long double f51

    cdef fit_coefficients_t *coeffs = <fit_coefficients_t *>malloc(sizeof(fit_coefficients_t))
    # compute auxiliary variables
    _fits_setup(coeffs, m1, m2, chi1, chi2)

    # initialise the coefficients for either versions
    if version == 1:
    
        # rational-function Pade coefficients (exact) from Eq. (22) of 1611.00332v1
        b10 = 0.487
        b20 = 0.295
        b30 = 0.17
        b50 = -0.0717

        # fit coefficients from Tables VII-X of 1611.00332v1
        # values at increased numerical precision copied from
        # https://git.ligo.org/uib-papers/finalstate2016/blob/master/LALInference/EradUIB2016_pyform_coeffs.txt
        # git commit 7b47e0f35a8f960b99b24caf3ffea2ddefdc4e29
        a2  = 0.5635376058169299
        a3  = -0.8661680065959881
        a4  = 3.181941595301782
        b1  = -0.15800074104558132
        b2  = -0.15815904609933157
        b3  = -0.14299315232521553
        b5  = 8.908772171776285
        f20 = 3.8071100104582234
        f30 = 25.99956516423936
        f50 = 1.552929335555098
        f10 = 1.7004558922558886
        f21 = 0.
        d10 = -0.12282040108157262
        d11 = -3.499874245551208
        d20 = 0.014200035799803777
        d30 = -0.01873720734635449
        d31 = -5.1830734185518725
        f11 = 14.39323998088354
        f31 = -232.25752840151296
        f51 = -0.8427987782523847

    elif version == 2:
    
        # rational-function Pade coefficients (exact) from Eq. (22) of 1611.00332v2
        b10 = 0.346
        b20 = 0.211
        b30 = 0.128
        b50 = -0.212
        
        # fit coefficients from Tables VII-X of 1611.00332v2
        # values at increased numerical precision copied from
        # https://git.ligo.org/uib-papers/finalstate2016/blob/master/LALInference/EradUIB2016v2_pyform_coeffs.txt
        # git commit f490774d3593adff5bb09ae26b7efc6deab76a42
        a2  = 0.5609904135313374
        a3  = -0.84667563764404
        a4  = 3.145145224278187
        b1  = -0.2091189048177395
        b2  = -0.19709136361080587
        b3  = -0.1588185739358418
        b5  = 2.9852925538232014
        f20 = 4.271313308472851
        f30 = 31.08987570280556
        f50 = 1.5673498395263061
        f10 = 1.8083565298668276
        f21 = 0.
        d10 = -0.09803730445895877
        d11 = -3.2283713377939134
        d20 = 0.01118530335431078
        d30 = -0.01978238971523653
        d31 = -4.91667749015812
        f11 = 15.738082204419655
        f31 = -243.6299258830685
        f51 = -0.5808669012986468

    else:
        raise ValueError('Unknown version -- should be either "v1" or "v2".')

    # calculate the radiated-energy fit from Eq. (27) of 1611.00332
    cdef long double Erad = (((1. + -2.0/3.0*coeffs.sqrt2)*coeffs.eta + a2*coeffs.eta2 + a3*coeffs.eta3 + a4*coeffs.eta4)*(1. + b10*b1*coeffs.Shat*(f10 + f11*coeffs.eta + (16. - 16.*f10 - 4.*f11)*coeffs.eta2) + b20*b2*coeffs.Shat2*(f20 + f21*coeffs.eta + (16. - 16.*f20 - 4.*f21)*coeffs.eta2) + b30*b3*coeffs.Shat3*(f30 + f31*coeffs.eta + (16. - 16.*f30 - 4.*f31)*coeffs.eta2)))/(1. + b50*b5*coeffs.Shat*(f50 + f51*coeffs.eta + (16. - 16.*f50 - 4.*f51)*coeffs.eta2)) + d10*coeffs.sqrt1m4eta*coeffs.eta2*(1. + d11*coeffs.eta)*coeffs.chidiff + d30*coeffs.Shat*coeffs.sqrt1m4eta*coeffs.eta*(1. + d31*coeffs.eta)*coeffs.chidiff + d20*coeffs.eta3*coeffs.chidiff2

    # convert to actual final mass
    cdef long double Mf = coeffs.m*(1.-Erad)
    free(coeffs)
    return Mf

def final_spin(long double m1, long double m2, long double chi1, long double chi2, unsigned int version=2):

    return _final_spin(m1, m2, chi1, chi2, version)

cdef long double _final_spin(long double m1, long double m2, long double chi1, long double chi2, unsigned int version) nogil:
    """
    | Calculates the final spin with the aligned-spin NR fit
     by `Xisco Jimenez Forteza, David Keitel, Sascha Husa et al. <https://arxiv.org/abs/1611.00332>`_
    | Versions :math:`v_1` and :math:`v_2` use the same ansatz, with :math:`v_2` calibrated to additional SXS and RIT data

    :param m1: primary mass in units of solar masses, :math:`m_1\\geq m_2`
    :type m1: float
    :param m2: secondary mass in units of solar masses
    :type m2: float
    :param chi1: dimensionless spin of the primary mass, :math:`z` component
    :type chi1: float
    :param chi2: dimensionless spin of the secondary mass, :math:`z` component
    :type chi2: float
    :param version: fit version, either :math:`v_1` or :math:`v_2`. *Optional*, default is :math:`v_2`
    :type version: string
    
    :return: final dimensionless spin :math:`\\chi_f`
    :rtype: float
    """

    cdef long double a2
    cdef long double a3
    cdef long double a4
    cdef long double b1
    cdef long double b2
    cdef long double b3
    cdef long double b5
    cdef long double f20
    cdef long double f30
    cdef long double f50
    cdef long double f10
    cdef long double f21
    cdef long double d10
    cdef long double d11
    cdef long double d20
    cdef long double d30
    cdef long double d31
    cdef long double f11
    cdef long double f31
    cdef long double f51

    cdef fit_coefficients_t *coeffs = <fit_coefficients_t *>malloc(sizeof(fit_coefficients_t))

    # compute auxiliary variables
    _fits_setup(coeffs, m1, m2, chi1, chi2)

    # initialise the coefficients for either versions
    if version == 1:
    
        # rational-function Pade coefficients (exact) from Eqs. (7) and (8) of 1611.00332v1
        a20 = 5.28
        a30 = 1.27
        a50 = 2.89
        b10 = -0.194
        b20 = 0.075
        b30 = 0.00782
        b50 = -0.527
        
        # fit coefficients from Tables I-IV of 1611.00332v1
        # evalues at increased numerical precision copied from
        # https://git.ligo.org/uib-papers/finalstate2016/blob/master/LALInference/FinalSpinUIB2016_pyform_coeffs.txt
        # git commit 7b47e0f35a8f960b99b24caf3ffea2ddefdc4e29
        a2  = 3.772362507208651
        a3  = -9.627812453422376
        a5  = 2.487406038123681
        b1  = 1.0005294518146604
        b2  = 0.8823439288807416
        b3  = 0.7612809461506448
        b5  = 0.9139185906568779
        f21 = 8.887933111404559
        f31 = 23.927104476660883
        f50 = 1.8981657997557002
        f11 = 4.411041530972546
        f52 = 0.
        d10 = 0.2762804043166152
        d11 = 11.56198469592321
        d20 = -0.05975750218477118
        d30 = 2.7296903488918436
        d31 = -3.388285154747212
        f12 = 0.3642180211450878
        f22 = -40.35359764942015
        f32 = -178.7813942566548
        f51 = -5.556957394513334

    elif version == 2:
    
        # rational-function Pade coefficients (exact) from Eqs. (7) and (8) of 1611.00332v2
        a20 = 5.24
        a30 = 1.3
        a50 = 2.88
        b10 = -0.194
        b20 = 0.0851
        b30 = 0.00954
        b50 = -0.579
        
        # fit coefficients from Tables I-IV of 1611.00332v2
        # values at increased numerical precision copied from
        # https://git.ligo.org/uib-papers/finalstate2016/blob/master/LALInference/FinalSpinUIB2016v2_pyform_coeffs.txt
        # git commit f490774d3593adff5bb09ae26b7efc6deab76a42
        a2  = 3.8326341618708577
        a3  = -9.487364155598392
        a5  = 2.5134875145648374
        b1  = 1.0009563702914628
        b2  = 0.7877509372255369
        b3  = 0.6540138407185817
        b5  = 0.8396665722805308
        f21 = 8.77367320110712
        f31 = 22.830033250479833
        f50 = 1.8804718791591157
        f11 = 4.409160174224525
        f52 = 0.
        d10 = 0.3223660562764661
        d11 = 9.332575956437443
        d20 = -0.059808322561702126
        d30 = 2.3170397514509933
        d31 = -3.2624649875884852
        f12 = 0.5118334706832706
        f22 = -32.060648277652994
        f32 = -153.83722669033995
        f51 = -4.770246856212403

    else:
        raise ValueError('Unknown version -- should be either "v1" or "v2".')

    # calculate the fit for the Lorb' quantity from Eq. (16) of 1611.00332
    cdef long double Lorb = (2.*coeffs.sqrt3*coeffs.eta + a20*a2*coeffs.eta2 + a30*a3*coeffs.eta3)/(1. + a50*a5*coeffs.eta) + (b10*b1*coeffs.Shat*(f11*coeffs.eta + f12*coeffs.eta2 + (64. - 16.*f11 - 4.*f12)*coeffs.eta3) + b20*b2*coeffs.Shat2*(f21*coeffs.eta + f22*coeffs.eta2 + (64. - 16.*f21 - 4.*f22)*coeffs.eta3) + b30*b3*coeffs.Shat3*(f31*coeffs.eta + f32*coeffs.eta2 + (64. - 16.*f31 - 4.*f32)*coeffs.eta3))/(1. + b50*b5*coeffs.Shat*(f50 + f51*coeffs.eta + f52*coeffs.eta2 + (64. - 64.*f50 - 16.*f51 - 4.*f52)*coeffs.eta3)) + d10*coeffs.sqrt1m4eta*coeffs.eta2*(1. + d11*coeffs.eta)*coeffs.chidiff + d30*coeffs.Shat*coeffs.sqrt1m4eta*coeffs.eta3*(1. + d31*coeffs.eta)*coeffs.chidiff + d20*coeffs.eta3*coeffs.chidiff2

    # convert to actual final spin, Stot and Lorb were adimensional, so it is correct, simply M_in_tot=1
    cdef long double chif = Lorb + coeffs.Stot
    free(coeffs)
    return chif
