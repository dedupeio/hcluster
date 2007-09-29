/**
 * cluster.c
 *
 * Author: Damian Eads
 * Date:   September 22, 2007
 *
 * Copyright (c) 2007, Damian Eads
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

#define NCHOOSE2(_n) ((_n)*(_n-1)/2)
#define ISCLUSTER(_nd) ((_nd)->id >= n)
#define GETCLUSTER(_id) ((lists + _id - n))

#define CPY_MAX(_x, _y) ((_x > _y) ? (_x) : (_y))
#define CPY_MIN(_x, _y) ((_x < _y) ? (_x) : (_y))


#include <malloc.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "cluster.h"

double euclidean_distance(const double *u, const double *v, int n) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    d = u[i] - v[i];
    s = s + d * d;
  }
  return sqrt(s);
}

double chebyshev_distance(const double *u, const double *v, int n) {
  int i = 0;
  double d, maxv = 0.0;
  for (i = 0; i < n; i++) {
    d = fabs(u[i] - v[i]);
    if (d > maxv) {
      maxv = d;
    }
  }
  return maxv;
}

double mahalanobis_distance(const double *u, const double *v, int n) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    d = fabs(u[i] - v[i]);
    s = s + d;
  }
  return sqrt(s);
}

double hamming_distance(const double *u, const double *v, int n) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    s = s + (u[i] != v[i]);
  }
  return s / (double)n;
}

double hamming_distance_bool(const char *u, const char *v, int n) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    s = s + (u[i] != v[i]);
  }
  return s / (double)n;
}

double jaccard_distance(const double *u, const double *v, int n) {
  int i = 0;
  double denom = 0.0, num = 0.0;
  for (i = 0; i < n; i++) {
    num += (u[i] != v[i]) && (u[i] != 0) || (v[i] != 0);
    denom += (u[i] != 0) || (v[i] != 0);
  }
  return num / denom;
}

double jaccard_distance_bool(const char *u, const char *v, int n) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    s = s + (u[i] != v[i]);
  }
  return s / (double)n;
}

double dot_product(const double *u, const double *v, int n) {
  int i;
  double s = 0.0;
  for (i = 0; i < n; i++) {
    s += u[i] * v[i];
  }
  return s;
}

double dot_product_sub(const double *u, double ub,
		       const double *v, double vb, int n) {
  int i;
  double s = 0.0;
  for (i = 0; i < n; i++) {
    s = s + ((u[i] - ub) * (v[i] - vb));
  }
  return s;
}

double vector_2norm(const double *u, int n) {
  return sqrt(dot_product(u, u, n));
}

double vector_mean(const double *u, int n) {
  int i;
  double s = 0.0;
  for (i = 0; i < n; i++) {
    s = s + u[i];
  }
  return s / (double)n;
}

double cosine_distance(const double *u, const double *v, int n,
		       const double nu, const double nv) {
  return 1.0 - (dot_product(u, v, n) / (nu * nv));
}

double seuclidean_distance(const double *var,
			   const double *u, const double *v, int n) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    d = u[i] - v[i];
    s = s + (d * d) / var[i];
  }
  return sqrt(s);
}

double city_block_distance(const double *u, const double *v, int n) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    d = fabs(u[i] - v[i]);
    s = s + d;
  }
  return s;
}

double minkowski_distance(const double *u, const double *v, int n, double p) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    d = fabs(u[i] - v[i]);
    s = s + pow(d, p);
  }
  return pow(s, 1.0 / p);
}

void compute_mean_vector(double *res, const double *X, int m, int n) {
  int i, j;
  const double *v;
  for (i = 0; i < n; i++) {
    res[i] = 0.0;
  }
  for (j = 0; j < m; j++) {

    v = X + (j * n);
    for (i = 0; i < n; i++) {
      res[i] += v[i];
    }
  }
  for (i = 0; i < n; i++) {
    res[i] /= (double)m;
  }
}

void vector_subtract(double *result, const double *u, const double *v, int n) {
  int i;
  for (i = 0; i < n; i++) {
    result[i] = u[i] - v[i];
  }
}

void pdist_euclidean(const double *X, double *dm, int m, int n) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = euclidean_distance(u, v, n);
    }
  }
}

void pdist_hamming(const double *X, double *dm, int m, int n) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = hamming_distance(u, v, n);
    }
  }
}

void pdist_hamming_bool(const char *X, double *dm, int m, int n) {
  int i, j;
  const char *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = hamming_distance_bool(u, v, n);
    }
  }
}

void pdist_jaccard(const double *X, double *dm, int m, int n) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = jaccard_distance(u, v, n);
    }
  }
}

void pdist_jaccard_bool(const char *X, double *dm, int m, int n) {
  int i, j;
  const char *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = jaccard_distance_bool(u, v, n);
    }
  }
}


void pdist_chebyshev(const double *X, double *dm, int m, int n) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = chebyshev_distance(u, v, n);
    }
  }
}

void pdist_cosine(const double *X, double *dm, int m, int n, const double *norms) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = cosine_distance(u, v, n, norms[i], norms[j]);
    }
  }
}

void pdist_seuclidean(const double *X, const double *var,
		     double *dm, int m, int n) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = seuclidean_distance(var, u, v, n);
    }
  }
}

void pdist_city_block(const double *X, double *dm, int m, int n) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = city_block_distance(u, v, n);
    }
  }
}

void pdist_minkowski(const double *X, double *dm, int m, int n, double p) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = minkowski_distance(u, v, n, p);
    }
  }
}


void chopmins(int *ind, int mini, int minj, int np) {
  int i;
  /**  if (mini < np - 2) {**/
  for (i = mini; i < minj - 1; i++) {
    ind[i] = ind[i + 1];
  }
  /**}**/
  /**  if (minj < np - 2) {**/
  for (i = minj - 1; i < np - 2; i++) {
    ind[i] = ind[i + 2];
  }
  /**  }**/
  /**  if (np > 0) {
    ind[np - 1] = HUGE_VALF;
  }
  if (np > 1) {
    ind[np - 2] = INFINITY;
    }***/
  /**  fprintf(stderr, "[Remove mini=%d minj=%d]\n", mini, minj);**/
}

void chopmins_ns_ij(double *ind, int mini, int minj, int np) {
  int i;
  /**if (mini < np - 2) {**/
  for (i = mini; i < minj - 1; i++) {
    ind[i] = ind[i + 1];
  }
  /**}**/
  /**if (minj < np - 2) {**/
  for (i = minj - 1; i < np - 2; i++) {
    ind[i] = ind[i + 2];
  }
  /**}**/
  /**  if (np > 0) {
    ind[np - 1] = INFINITY;
  }
  if (np > 1) {
    ind[np - 2] = INFINITY;
    }**/
}

void chopmins_ns_i(double *ind, int mini, int np) {
  int i;
    for (i = mini; i < np - 1; i++) {
      ind[i] = ind[i + 1];
    }
    /**  if (np > 0) {
    ind[np - 1] = INFINITY;
    }**/
}

void dist_single(cinfo *info, int mini, int minj, int np, int n) {
  double **rows = info->rows;
  double *buf = info->buf;
  double *bit;
  int i;
  bit = buf;
  for (i = 0; i < mini; i++, bit++) {
    *bit = CPY_MIN(*(rows[i] + mini - i - 1), *(rows[i] + minj - i - 1));
  }
  for (i = mini + 1; i < minj; i++, bit++) {
    *bit = CPY_MIN(*(rows[mini] + i - mini - 1), *(rows[i] + minj - i - 1));
  }
  for (i = minj + 1; i < np; i++, bit++) {
    *bit = CPY_MIN(*(rows[mini] + i - mini - 1), *(rows[minj] + i - minj - 1));
  }
  /**  fprintf(stderr, "[");
  for (i = 0; i < np - 2; i++) {
    fprintf(stderr, "%5.5f ", buf[i]);
  }
  fprintf(stderr, "]");**/
}

void dist_complete(cinfo *info, int mini, int minj, int np, int n) {
  double **rows = info->rows;
  double *buf = info->buf;
  double *bit;
  int i;
  bit = buf;
  for (i = 0; i < mini; i++, bit++) {
    *bit = CPY_MAX(*(rows[i] + mini - i - 1), *(rows[i] + minj - i - 1));
  }
  for (i = mini + 1; i < minj; i++, bit++) {
    *bit = CPY_MAX(*(rows[mini] + i - mini - 1), *(rows[i] + minj - i - 1));
  }
  for (i = minj + 1; i < np; i++, bit++) {
    *bit = CPY_MAX(*(rows[mini] + i - mini - 1), *(rows[minj] + i - minj - 1));
  }
}

void dist_average(cinfo *info, int mini, int minj, int np, int n) {
  double **rows = info->rows, *buf = info->buf, *bit;
  int *inds = info->ind;
  double drx, dsx, denom, mply, rscnt, rc, sc;
  int i, xi, xn;
  cnode *rn = info->nodes + inds[mini];
  cnode *sn = info->nodes + inds[minj];
  bit = buf;
  rc = (double)rn->n;
  sc = (double)sn->n;
  rscnt = rc + sc;

  for (i = 0; i < mini; i++, bit++) {
    /** d(r,x) **/
    drx = *(rows[i] + mini - i - 1);
    dsx = *(rows[i] + minj - i - 1);
    xi = inds[i];
    cnode *xnd = info->nodes + xi;
    xn = xnd->n;
    mply = 1.0 / (((double)xn) * rscnt);
    *bit = mply * ((drx * (rc * xn)) + (dsx * (sc * xn)));
  }
  for (i = mini + 1; i < minj; i++, bit++) {
    drx = *(rows[mini] + i - mini - 1);
    dsx = *(rows[i] + minj - i - 1);
    xi = inds[i];
    cnode *xnd = info->nodes + xi;
    xn = xnd->n;
    mply = 1.0 / (((double)xn) * rscnt);
    *bit = mply * ((drx * (rc * xn)) + (dsx * (sc * xn)));
  }
  for (i = minj + 1; i < np; i++, bit++) {
    drx = *(rows[mini] + i - mini - 1);
    dsx = *(rows[minj] + i - minj - 1);
    xi = inds[i];
    cnode *xnd = info->nodes + xi;
    xn = xnd->n;
    mply = 1.0 / (((double)xn) * rscnt);
    *bit = mply * ((drx * (rc * xn)) + (dsx * (sc * xn)));
  }
}

void dist_centroid(cinfo *info, int mini, int minj, int np, int n) {
  double **rows = info->rows, *buf = info->buf, *bit;
  int *inds = info->ind;
  const double *centroid_tq;
  int i, m, xi;
  centroid_tq = info->centroids[info->nid];
  bit = buf;
  m = info->m;
  for (i = 0; i < np; i++, bit++) {
    /** d(r,x) **/
    if (i == mini || i == minj) {
      bit--;
      continue;
    }
    xi = inds[i];
    *bit = euclidean_distance(info->centroids[xi], centroid_tq, m);
    /**    fprintf(stderr, "%5.5f ", *bit);**/
  }
  /**  fprintf(stderr, "\n");**/
}

void dist_ward(cinfo *info, int mini, int minj, int np, int n) {
  double **rows = info->rows, *buf = info->buf, *bit;
  int *inds = info->ind;
  const double *centroid_tq;
  int i, m, xi, xn, rind, sind;
  double rn, sn, qn, num;
  cnode *left, *right;
  rind = inds[mini];
  sind = inds[minj];
  rn = (double)info->nodes[rind].n;
  sn = (double)info->nodes[sind].n;
  centroid_tq = info->centroids[info->nid];
  bit = buf;
  m = info->m;
  for (i = 0; i < np; i++, bit++) {
    /** d(r,x) **/
    if (i == mini || i == minj) {
      bit--;
      continue;
    }
    xi = inds[i];
    qn = (double)info->nodes[xi].n;
    num = euclidean_distance(info->centroids[xi], centroid_tq, m);
    *bit = sqrt(((rn + sn) * qn) * ((num * num) / (qn + rn + sn)));
    /**    fprintf(stderr, "%5.5f ", *bit);**/
  }
  /**  fprintf(stderr, "\n");**/
}

void print_dm(const double **rows, int np) {
  int i, j, k;
  const double *row;
  fprintf(stderr, "[DM, np=%d\n", np);
  for (i = 0; i < np - 1; i++) {
    row = rows[i];
    for (j = 0; j <= i; j++) {
      fprintf(stderr, "%5.5f ", 0.0);
    }

    for (k = 0, j = i + 1; j < np; j++, k++) {
      fprintf(stderr, "%5.5f ", *(row + k));
    }
    fprintf(stderr, "|j=%d|\n", i + 1);
  }
}

void print_ind(const int *inds, int np) {
  int i;
  fprintf(stderr, "[IND, np=%d || ", np);
  for (i = 0; i < np; i++) {
    fprintf(stderr, "%d ", inds[i]);
  }
  fprintf(stderr, "]\n");
}

void print_vec(const double *d, int n) {
  int i;
  fprintf(stderr, "[");
  for (i = 0; i < n; i++) {
    fprintf(stderr, "%5.5f ", d[i]);
  }
  fprintf(stderr, "]");
}

/**
 * notes to self:
 * dm:    The distance matrix.
 * Z:     The result of the linkage, a (n-1) x 3 matrix.
 * X:     The original observations as row vectors (=NULL if not needed).
 * n:     The number of objects.
 * ml:    A boolean indicating whether a list of objects in the forest
 *        clusters should be maintained.
 * kc:    Keep track of the centroids.
 */
void linkage(double *dm, double *Z, const double *X,
	     int m, int n, int ml, int kc, distfunc dfunc,
	     int method) {
  int i, j, k, t, np, nid, mini, minj;
  double min, ln, rn, qn;
  int *ind;
  /** An iterator through the distance matrix. */
  double *dmit, *buf;

  int *rowsize;

  /** Temporary array to store modified distance matrix. */
  double *dmt, **rows;
  double *centroidsData;
  double **centroids;
  const double *centroidL, *centroidR;
  double *centroid;
  clist *lists, *listL, *listR, *listC;
  clnode *lnodes;
  cnode *nodes, *node;

  cinfo info;


  /** The next two are only necessary for euclidean distance methods. */
  if (ml) {
    lists = (clist*)malloc(sizeof(clist) * (n-1));
    lnodes = (clnode*)malloc(sizeof(clnode) * n);
  }
  else {
    lists = 0;
    lnodes = 0;
  }
  if (kc) {
    centroids = (double**)malloc(sizeof(double*) * (2 * n));
    centroidsData = (double*)malloc(sizeof(double) * n * m);
    for (i = 0; i < n; i++) {
      centroids[i] = X + i * m;
    }
    for (i = 0; i < n; i++) {
      centroids[i+n] = centroidsData + i * m;
    }
  }
  else {
    centroids = 0;
    centroidsData = 0;
  }

  nodes = (cnode*)malloc(sizeof(cnode) * (n * 2) - 1);
  ind = (int*)malloc(sizeof(int) * n);
  dmt = (double*)malloc(sizeof(double) * NCHOOSE2(n));
  buf = (double*)malloc(sizeof(double) * n);
  rows = (double**)malloc(sizeof(double*) * n);
  rowsize = (int*)malloc(sizeof(int) * n);
  memcpy(dmt, dm, sizeof(double) * NCHOOSE2(n));

  info.X = X;
  info.m = m;
  info.n = n;
  info.nodes = nodes;
  info.ind = ind;
  info.dmt = dmt;
  info.buf = buf;
  info.rows = rows;
  info.rowsize = rowsize;
  info.dm = dm;
  info.centroids = centroids;

  for (i = 0; i < n; i++) {
    ind[i] = i;
    node = nodes + i;
    node->left = 0;
    node->right = 0;
    node->id = i;
    node->n = 1;
    node->d = 0.0;
    rowsize[i] = n - 1 - i;
  }
  rows[0] = dmt;
  for (i = 1; i < n; i++) {
    rows[i] = rows[i-1] + n - i;
  }
  
  if (ml) {
    for (i = 0; i < n; i++) {
      (lnodes + i)->val = nodes + i;
      (lnodes + i)->next = 0;
    }
  }

  for (k = 0, nid = n; k < n - 1; k++, nid++) {
    info.nid = nid;
    np = n - k;
    /**    fprintf(stderr, "k=%d, nid=%d, n=%d np=%d\n", k, nid, n, np);**/
    min = dmt[0];
    mini = 0;
    minj = 1;
    /** Note that mini < minj since j > i is always true. */
    for (i = 0; i < np - 1; i++) {
      dmit = rows[i];
      for (j = i + 1; j < np; j++, dmit++) {
	if (*dmit < min) {
	  min = *dmit;
	  mini = i;
	  minj = j;
	}
      }
    }

    node = nodes + nid;
    node->left = nodes + ind[mini];
    node->right = nodes + ind[minj];
    ln = (double)node->left->n;
    rn = (double)node->right->n;
    qn = ln + rn;
    node->n = node->left->n + node->right->n;
    node->d = min;
    node->id = nid;

    *(Z + (k * 3)) = node->left->id;
    *(Z + (k * 3) + 1) = node->right->id;
    *(Z + (k * 3) + 2) = min;

    /**    fprintf(stderr,
	    "[lid=%d, rid=%d, llid=%d, rrid=%d m=%5.8f]",
	    node->left->id, node->right->id, ind[mini], ind[minj], min);**/

    if (kc) {
      centroidL = centroids[ind[mini]];
      centroidR = centroids[ind[minj]];
      centroid = centroids[nid];
      switch(method) {
      case CPY_LINKAGE_MEDIAN:
	for (t = 0; t < m; t++) {
	  centroid[t] = (centroidL[t] * 0.5 + centroidR[t] * 0.5);
	}
	break;
      case CPY_LINKAGE_CENTROID:
      case CPY_LINKAGE_WARD:
      default:
	for (t = 0; t < m; t++) {
	  centroid[t] = (centroidL[t] * ln + centroidR[t] * rn) / qn;
	}
	break;
      }
      /**      fprintf(stderr, "L: ");
      print_vec(centroidL, m);
      fprintf(stderr, "\nR: ");
      print_vec(centroidR, m);
      fprintf(stderr, "\nT: ");
      print_vec(centroid, m);**/
    }
    if (ml) {
      listC = GETCLUSTER(nid);
      if (ISCLUSTER(node->left) != 0) {
	listL = GETCLUSTER(node->left->id);
	if (ISCLUSTER(node->right) != 0) {
	  listR = GETCLUSTER(node->right->id);
	  listL->tail->next = listR->head;
	  listC->tail = listR->tail;
	  listR->tail->next = 0;
	}
	else {
	  listC->tail = lnodes + node->right->id;
	  listL->tail->next = listC->tail;
	  listC->tail->next = 0;
	}
	listC->head = listL->head;
      }
      else {
	listC->head = lnodes + node->left->id;
	if (ISCLUSTER(node->right)) {
	  listR = GETCLUSTER(node->right->id);
	  listC->head->next = listR->head;
	  listC->tail = listR->tail;
	  listC->tail->next = 0;
	}
	else {
	  listC->tail = lnodes + node->right->id;
	  listC->tail->next = 0;
	  listC->head->next = listC->tail;
	}
      }
    }
    /**    print_dm(rows, np);**/
    /**    dfunc(buf, rows, mini, minj, np, dm, n, ind, nodes);**/
    dfunc(&info, mini, minj, np, n);

    /** For these rows, we must remove, i and j but leave all unused space
        at the end. This reduces their size by two.*/
    for (i = 0; i < mini; i++) {
      chopmins_ns_ij(rows[i], mini - i - 1, minj - i - 1, rowsize[i]);
    }

    /** We skip the i'th row. For rows i+1 up to j-1, we just remove j. */
    for (i = mini + 1; i < minj; i++) {
      chopmins_ns_i(rows[i], minj - i - 1, rowsize[i]);
    }

    /** For rows 0 to mini - 1, we move them down the matrix, leaving the
	first row free. */
    /**    for (i = mini; i > 0; i--) {
      memcpy(rows[i], rows[i-1], sizeof(double) * rowsize[i]-k);
      }**/

    for (i = mini; i < minj - 1; i++) {
      memcpy(rows[i], rows[i+1], sizeof(double) * (rowsize[i+1]));
    }

    /** For rows mini+1 to minj-1, we do nothing since they are in the
	right place for the next iteration. For rows minj+1 onward,
	we move them to the right. */
	
    for (i = minj - 1; i < np - 2; i++) {
      memcpy(rows[i], rows[i+2], sizeof(double) * (rowsize[i+2]));
    }

    /** Rows i+1 to j-1 lose one unit of space, so we move them up. */
    /** Rows j to np-1 lose no space. We do nothing to them. */

    /**    memcpy(rows[0], buf, sizeof(double) * rowsize[0] - k);*/

    for (i = 0; i < np - 2; i++) {
      *(rows[i] + np - 3 - i) = buf[i];
    }
    /**    print_dm(rows, np - 1);
	   print_ind(ind, np);**/
    chopmins(ind, mini, minj, np);
    ind[np - 2] = nid;
    /**    print_ind(ind, np - 1);**/
  }
  free(lists);
  free(lnodes);
  free(nodes);
  free(ind);
  free(dmt);
  free(buf);
  free(rows);
  free(rowsize);
  free(centroidsData);
  free(centroids);
}

/**
 * endpnts is a (n-1) by 4 by 2 array
 *
 * ctrpnts is a (n-1) by 2 by 2 array
 *
 * edge is a (n-1) by 2 by 2 array
 *
 * sbl: size between leaves
 *
 * Thoughts to self: might be more efficient to compute these bits and
 * pieces when constructing z.
 */
/**
void get_dendrogram_line_endpoints(const double *Z,
				   int n,
				   double *endpnts,
				   double *ctrpnts
				   double *edge,
				   double sbl) {
  const double *Zit;
  double *endpntsit;
  double *ctrpntsit;
  double *edgeit;
  int i, j;
  for (i = 0; i < n - 1; i++) {
    Zit = Z + (4 * i)
  }
  }**/

/** Stub. **/
void compute_inconsistency_coefficient(const double *Z, double *Y, int d) {
  return;
}

void dist_to_squareform_from_vector(double *M, const double *v, int n) {
  double *it;
  const double *cit;
  int i, j;
  cit = v;
  for (i = 0; i < n - 1; i++) {
    it = M + (i * n) + i + 1;
    for (j = i + 1; j < n; j++, it++, cit++) {
      *it = *cit;
    }
  }
}

void dist_to_vector_from_squareform(const double *M, double *v, int n) {
  double *it;
  const double *cit;
  int i, j;
  it = v;
  for (i = 0; i < n - 1; i++) {
    cit = M + (i * n) + i + 1;
    for (j = i + 1; j < n; j++, it++, cit++) {
      *it = *cit;
    }
  }
}
