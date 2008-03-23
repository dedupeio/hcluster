/**
 * pdist.hpp
 *
 * Author: Damian Eads
 * Date:   September 22, 2007
 *
 * Copyright (c) 2007, 2008, Damian Eads
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   - Redistributions of source code must retain the above
 *     copyright notice, this list of conditions and the
 *     following disclaimer.
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer
 *     in the documentation and/or other materials provided with the
 *     distribution.
 *   - Neither the name of the author nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <string.h>
#include <math.h>

#ifndef RISOTTO_HCLUSTER_H
#define RISOTTO_HCLUSTER_H

namespace Risotto {

  namespace Hcluster {

    template <typename RT, typename IT> 
    class Pdist {
    
    public:
      Pdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) : dim(_dim),
								     num_vectors(_num_vectors) {}
    
      virtual RT distance(const IT *u, const IT *v) = 0;
    
      virtual void pairwise() {
	const IT *u, *v;
	RT *it = dm;
      
	for (i = 0; i < num_vectors; i++) {
	  for (j = i + 1; j < num_vectors; j++, it++) {
	    u = X + (dim * i);
	    v = X + (dim * j);
	    *it = distance(u, v);
	  }
	}
      }

      virtual size_t getDimension() const {
	return dim;
      }

    protected:
      size_t i;
      size_t j;
      const size_t dim;
      const size_t num_vectors;
      const IT *X;
      RT *dm;  
    };

    template <typename RT, typename IT>
    class BrayCurtisPdist : public Pdist <RT, IT> {

    public:
      BrayCurtisPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t i;
	RT s1 = 0.0, s2 = 0.0;
	for (i = 0; i < this->dim; i++) {
	  s1 += (RT)fabs(u[i] - v[i]);
	  s2 += (RT)fabs(u[i] + v[i]);
	}
	return s1 / s2;
      }  
    };


    template <typename RT, typename IT>
    class CanberraPdist : public Pdist <RT, IT> {

    public:
      CanberraPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t i;
	RT s = 0.0;
	for (i = 0; i < this->dim; i++) {
	  s += (fabs((RT)(u[i] - v[i])) / (fabs((RT)u[i]) + fabs((RT)v[i])));
	}
	return s;
      }
  
    };

    template <typename RT, typename IT>
    class ChebyshevPdist : public Pdist <RT, IT> {

    public:
      ChebyshevPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t i = 0;
	RT d, maxv = 0.0;
	for (i = 0; i < this->dim; i++) {
	  d = fabs((IT)(u[i] - v[i]));
	  if (d > maxv) {
	    maxv = d;
	  }
	}
	return maxv;
      }  
    };

    template <typename RT, typename IT>
    class CityBlockPdist : public Pdist <RT, IT> {

    public:
      CityBlockPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	int i = 0;
	double s = 0.0, d;
	for (i = 0; i < this->dim; i++) {
	  d = fabs(u[i] - v[i]);
	  s = s + d;
	}
	return s;
      }
    };

    template <typename RT, typename IT>
    static RT dot(const IT *u, const IT *v, int n) {
      size_t i;
      long double s = 0.0;
      for (i = 0; i < n; i++) {
	s += (long double)(u[i] * v[i]);
      }
      return (RT)s;
    }

    template <typename RT, typename IT, typename NORMT>
    class CosinePdist : public Pdist <RT, IT> {

    public:
      CosinePdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors, const NORMT *_norms) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors), norms(_norms) {}

      virtual RT distance(const IT *u, const IT *v) {
	return 1.0 - (dot<RT, IT>(u, v, this->dim) / (norms[this->i] * norms[this->j]));
      }

    protected:
      NORMT *norms;

    };


    template <typename RT, typename IT>
    class DicePdist: public Pdist <RT, IT> {

    public:
      DicePdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t i = 0;
	size_t ntt = 0, nft = 0, ntf = 0;
	for (i = 0; i < this->dim; i++) {
	  ntt += (u[i] && v[i]);
	  ntf += (u[i] && !v[i]);
	  nft += (!u[i] && v[i]);
	}
	return (RT)((long double)(nft + ntf) / (long double)(2.0 * ntt + ntf + nft));
      }
    };

    template <typename RT, typename IT>
    class EuclideanPdist : public Pdist <RT, IT> {

    public:
      EuclideanPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t i = 0;
	RT s = 0.0, d;
	for (i = 0; i < this->dim; i++) {
	  d = (RT)(u[i] - v[i]);
	  s = s + d * d;
	}
	return sqrt(s);
      }  
    };

    template <typename RT, typename IT>
    class HammingPdist : public Pdist <RT, IT> {

    public:
      HammingPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t i = 0;
	RT s = 0.0;
	for (i = 0; i < this->dim; i++) {
	  s = s + (u[i] != v[i]);
	}
	return s / (RT)this->dim;
      }
    };

    template <typename RT, typename IT>
    class JaccardPdist: public Pdist <RT, IT> {

    public:
      JaccardPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	int i = 0;
	double denom = 0.0, num = 0.0;
	for (i = 0; i < this->dim; i++) {
	  num += (u[i] != v[i]) && ((u[i] != 0) || (v[i] != 0));
	  denom += (u[i] != 0) || (!v[i] != 0);
	}
	return num / denom;
      }
    };

    /**
       template <typename RT>
       class JaccardPdist<RT, bool> {

       public:
       JaccardPdist(RT *dm, const bool *X, size_t _dim, size_t _num_vectors) :
       Pdist<RT, bool>(dm, X, _dim, _num_vectors) {}

       virtual RT distance(const bool *u, const bool *v) {
       int i = 0;
       long double s = 0.0;
       for (i = 0; i < this->dim; i++) {
       s = s + (u[i] != v[i]);
       }
       return s / (long double)this->dim;

       }
       };

       JaccardPdist<double, bool> foo((double*)0, (const bool*)0, 5, 5);
    **/


    template <typename RT, typename IT>
    class KulsinskiPdist : public Pdist <RT, IT> {

    public:
      KulsinskiPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t _i = 0;
	size_t ntt = 0, nft = 0, ntf = 0, nff = 0;
	for (_i = 0; _i < this->n; _i++) {
	  ntt += (u[_i] && v[_i]);
	  ntf += (u[_i] && !v[_i]);
	  nft += (!u[_i] && v[_i]);
	  nff += (!u[_i] && !v[_i]);
	}
	return (RT)((long double)(ntf + nft - ntt + this->dim)) / ((long double)(ntf + nft + this->dim));
      }
    };

    template <typename RT, typename IT>
    class MatchingPdist: public Pdist <RT, IT> {

    public:
      MatchingPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t i = 0;
	size_t ntt = 0, nff = 0, nft = 0, ntf = 0;
	for (i = 0; i < this->dim; i++) {
	  ntt += (u[i] && v[i]);
	  ntf += (u[i] && !v[i]);
	  nft += (!u[i] && v[i]);
	  nff += (!u[i] && !v[i]);
	}
	return (RT)((long double)(2.0 * ntf * nft) / (long double)(ntt * nff + ntf * nft)); 
      }
    };


    template <typename RT, typename IT, typename COVT>
    class MahalanobisPdist : public Pdist <RT, IT> {

    public:
      MahalanobisPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors,
		       const COVT *_covinv) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors), dimbuf1(0), dimbuf2(0), covinv(_covinv) {
	dimbuf1 = new RT[sizeof(double) * _dim];
	dimbuf2 = new RT[sizeof(double) * _dim];
      }

      ~MahalanobisPdist() {
	delete []dimbuf1;
	delete []dimbuf2;
      }
  
      virtual RT distance(const IT *u, const IT *v) {
	size_t i, j;
	double s;
	const size_t dim(this->dim);
	const COVT *covrow = covinv;
	for (i = 0; i < dim; i++) {
	  dimbuf1[i] = (RT)(u[i] - v[i]);
	}
	for (i = 0; i < dim; i++) {
	  covrow = covinv + (i * dim);
	  s = 0.0;
	  for (j = 0; j < dim; j++) {
	    s += dimbuf1[j] * (RT)covrow[j];
	  }
	  dimbuf2[i] = s;
	}
	s = 0.0;
	for (i = 0; i < this->dim; i++) {
	  s += dimbuf1[i] * dimbuf2[i];
	}
	return sqrt(s);
      }

    public:
      RT *dimbuf1;
      RT *dimbuf2;
      COVT *covinv;
    };

    template <typename RT, typename IT>
    class MinkowskiPdist: public Pdist <RT, IT> {

    public:
      MinkowskiPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors, long double _p) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors), p(_p) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t i = 0;
	double s = 0.0, d;
	for (i = 0; i < this->dim; i++) {
	  d = fabsl(u[i] - v[i]);
	  s = s + pow(d, p);
	}
	return powl(s, 1.0 / p);
      }

    protected:
      long double p;
    };

    template <typename RT, typename IT>
    class RogersTanimotoPdist: public Pdist <RT, IT> {

    public:
      RogersTanimotoPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t i = 0;
	size_t ntt = 0, nff = 0, nft = 0, ntf = 0;
	for (i = 0; i < this->dim; i++) {
	  ntt += (u[i] && v[i]);
	  ntf += (u[i] && !v[i]);
	  nft += (!u[i] && v[i]);
	  nff += (!u[i] && !v[i]);
	}
	return (RT)((long double)(2.0 * (ntf + nft))) / ((long double)ntt + nff + (2.0 * (ntf + nft)));
      }
    };

    template <typename RT, typename IT>
    class RussellRaoPdist: public Pdist <RT, IT> {

    public:
      RussellRaoPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t i = 0;
	size_t ntt = 0;
	for (i = 0; i < this->dim; i++) {
	  ntt += (u[i] && v[i]);
	}
	return (RT)((long double) (this->dim - ntt) / (long double) this->dim);
      }
    };

    template <typename RT, typename IT, typename VART>
    class SEuclideanPdist : public Pdist <RT, IT> {

    public:
      SEuclideanPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors, const VART *_var) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors), var(_var) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t i = 0;
	RT s = 0.0, d;
	for (i = 0; i < this->dim; i++) {
	  d = (RT)(u[i] - v[i]);
	  s = s + (RT)((long double)(d * d) / (long double)var[i]);
	}
	return sqrt(s);
      }

    protected:
      VART *var;
    };

    template <typename RT, typename IT>
    class SokalMichenerPdist : public Pdist <RT, IT> {

    public:
      SokalMichenerPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t _i = 0;
	size_t ntt = 0, nft = 0, ntf = 0, nff = 0;
	for (_i = 0; _i < this->dim; _i++) {
	  ntt += (u[_i] && v[_i]);
	  nff += (!u[_i] && !v[_i]);
	  ntf += (u[_i] && !v[_i]);
	  nft += (!u[_i] && v[_i]);
	}
	return (RT)((long double)(2.0 * (ntf + nft))/(long double)(2.0 * (ntf + nft) + ntt + nff));
      }
    };


    template <typename RT, typename IT>
    class SokalSneathPdist : public Pdist <RT, IT> {

    public:
      SokalSneathPdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t _i = 0;
	size_t ntt = 0, nft = 0, ntf = 0;
	for (_i = 0; _i < this->dim; _i++) {
	  ntt += (u[_i] && v[_i]);
	  ntf += (u[_i] && !v[_i]);
	  nft += (!u[_i] && v[_i]);
	}
	return (RT)((long double)(2.0 * (ntf + nft))/(long double)(2.0 * (ntf + nft) + ntt));
      }
    };

    template <typename RT, typename IT>
    class YulePdist: public Pdist <RT, IT> {

    public:
      YulePdist(RT *dm, const IT *X, size_t _dim, size_t _num_vectors) :
	Pdist<RT, IT>(dm, X, _dim, _num_vectors) {}

      virtual RT distance(const IT *u, const IT *v) {
	size_t i = 0;
	size_t ntt = 0, nff = 0, nft = 0, ntf = 0;
	for (i = 0; i < this->dim; i++) {
	  ntt += (u[i] && v[i]);
	  ntf += (u[i] && !v[i]);
	  nft += (!u[i] && v[i]);
	  nff += (!u[i] && !v[i]);
	}
	return (RT)((long double)(2.0 * ntf * nft) / (long double)(ntt * nff + ntf * nft)); 
      }
    };
  }
}

#endif
