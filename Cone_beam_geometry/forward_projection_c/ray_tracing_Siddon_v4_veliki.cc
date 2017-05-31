#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

using namespace std;

#define NX 1024
#define NY 1024
#define NZ 1024
#define MAXZ_SLICE 180
#define NP 1076
#define NC (2*138)
#define NR (2*16)

double pi = acos(-1.0);

double eps = 1e-7;
bool lt(double a, double b) { return a + eps < b; }
bool eq(double a, double b) { return !lt(a, b) && !lt(b, a); }
void linspace(double x1, double x2, int N, double* result) {
  for (int i = 0; i < N; ++i) {
    double f = 1.0 * i / (N - 1);
    result[i] = x1 * (1 - f) + x2 * f;
  }
}

void arange(double x1, double x2, int N, double dx, double* result) {
  for (int i = 0; i < N; ++i) {
    result[i] = x1 + dx * i;
  }
}

struct Point {
  double x, y, z;
  double& operator()(char dim) {
    if (dim == 'x') return x;
    if (dim == 'y') return y;
    if (dim == 'z') return z;
  }
  double operator()(char dim) const {
    if (dim == 'x') return x;
    if (dim == 'y') return y;
    if (dim == 'z') return z;
  }
};

Point operator+(Point const& a, Point const& b) {
  return Point{a.x + b.x, a.y + b.y, a.z + b.z};
}

Point operator*(double c, Point const& a) {
  return Point{c * a.x, c * a.y, c * a.z};
}

Point operator/(Point const& a, double b) {
  return Point{a.x / b, a.y / b, a.z / b};
}

double dot(Point const& a, Point const& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct IndexPoint {
  int x, y, z;
  int& operator()(char dim) {
    if (dim == 'x') return x;
    if (dim == 'y') return y;
    if (dim == 'z') return z;
  }
};

Point localGradient(double obj[NX][NY][NZ], IndexPoint const& vox_coor,
                    IndexPoint const& N, Point const& d) {
  Point grad;
  if (vox_coor.x + 1 < N.x && vox_coor.x - 1 >= 0) {
    grad.x = (obj[vox_coor.x + 1][vox_coor.y][vox_coor.z] -
              obj[vox_coor.x - 1][vox_coor.y][vox_coor.z]) / (2.0 * d.x);
  } else {
    if (vox_coor.x + 1 >= N.x) {
      grad.x = (obj[vox_coor.x][vox_coor.y][vox_coor.z] -
                obj[vox_coor.x - 1][vox_coor.y][vox_coor.z]) / (2.0 * d.x);
    } else {
      grad.x = (obj[vox_coor.x + 1][vox_coor.y][vox_coor.z] -
                obj[vox_coor.x][vox_coor.y][vox_coor.z]) / (2.0 * d.x);
    }
  }
  if (vox_coor.y + 1 < N.y && vox_coor.y - 1 >= 0) {
    grad.y = (obj[vox_coor.x][vox_coor.y + 1][vox_coor.z] -
              obj[vox_coor.x][vox_coor.y - 1][vox_coor.z]) / (2.0 * d.y);
  } else {
    if (vox_coor.y + 1 >= N.y) {
      grad.y = (obj[vox_coor.x][vox_coor.y][vox_coor.z] -
                obj[vox_coor.x][vox_coor.y - 1][vox_coor.z]) / (2.0 * d.y);
    } else {
      grad.y = (obj[vox_coor.x][vox_coor.y + 1][vox_coor.z] -
                obj[vox_coor.x][vox_coor.y][vox_coor.z]) / (2.0 * d.y);
    }
  }
  if (vox_coor.z + 1 < N.z && vox_coor.z - 1 >= 0) {
    grad.z = (obj[vox_coor.x][vox_coor.y][vox_coor.z + 1] -
              obj[vox_coor.x][vox_coor.y][vox_coor.z - 1]) / (2.0 * d.x);
  } else {
    if (vox_coor.z + 1 >= N.z) {
      grad.z = (obj[vox_coor.x][vox_coor.y][vox_coor.z] -
                obj[vox_coor.x][vox_coor.y][vox_coor.z - 1]) / (2.0 * d.x);
    } else {
      grad.z = (obj[vox_coor.x][vox_coor.y][vox_coor.z + 1] -
                obj[vox_coor.x][vox_coor.y][vox_coor.z]) / (2.0 * d.x);
    }
  }
  return grad;
}

double index2alpha(int i, double p1, double p2, double b, double d) {
  return ((b + 1.0*i*d) - p2)/(p1 - p2);
}

double alpha2index(double alpha, double p1, double p2, double b, double d) {
  return (p2 + alpha*(p1-p2) - b)/d;
}

Point get_point(double alpha, Point const& p1, Point const& p2) {
  return Point{p2.x + alpha * (p1.x - p2.x),
               p2.y + alpha * (p1.y - p2.y),
               p2.z + alpha * (p1.z - p2.z)};
}

struct MinMaxPlaneIndicesResult {
  int imin, imax;
  int jmin, jmax;
  int kmin, kmax;
  double alpha_min, alpha_max;
};

MinMaxPlaneIndicesResult min_max_plane_indices(
    IndexPoint const& N, Point const& p1, Point const& p2,
    Point const& b, Point const& d) {
  double alphax_min = min(index2alpha(N.x, p1.x, p2.x, b.x, d.x),
                          index2alpha(0, p1.x, p2.x, b.x, d.x));
  double alphax_max = max(index2alpha(N.x, p1.x, p2.x, b.x, d.x),
                          index2alpha(0, p1.x, p2.x, b.x, d.x));
  double alphay_min = min(index2alpha(N.y, p1.y, p2.y, b.y, d.y),
                          index2alpha(0, p1.y, p2.y, b.y, d.y));
  double alphay_max = max(index2alpha(N.y, p1.y, p2.y, b.y, d.y),
                          index2alpha(0, p1.y, p2.y, b.y, d.y));
  double alphaz_min = min(index2alpha(N.z, p1.z, p2.z, b.z, d.z),
                          index2alpha(0, p1.z, p2.z, b.z, d.z));
  double alphaz_max = max(index2alpha(N.z, p1.z, p2.z, b.z, d.z),
                          index2alpha(0, p1.z, p2.z, b.z, d.z));
  double alpha_min = max(alphax_min, alphay_min);
  double alpha_max = min(alphax_max, min(alphay_max, alphaz_max));
  int imin = -1;
  int jmin = -1;
  int kmin = -1;
  int imax = -1;
  int jmax = -1;
  int kmax = -1;
  if (alpha_max > alpha_min) {
    if (p1.x < p2.x) {
      if (eq(alpha_min, alphax_min)) {
        imin = N.x;
      } else {
        imin = floor(alpha2index(alpha_min, p1.x, p2.x, b.x, d.x));
      }
      if (eq(alpha_max, alphax_max)) {
        imax = 0;
      } else {
        imax = ceil(alpha2index(alpha_max, p1.x, p2.x, b.x, d.x));
      }
    } else if (p1.x > p2.x) {
      if (eq(alpha_max, alphax_max)) {
        imax = N.x;
      } else {
        imax = floor(alpha2index(alpha_max, p1.x, p2.x, b.x, d.x));
      }
      if (eq(alpha_min, alphax_min)) {
        imin = 0;
      } else {
        imin = ceil(alpha2index(alpha_min, p1.x, p2.x, b.x, d.x));
      }
    }

    if (p1.y < p2.y) {
      if (eq(alpha_min, alphay_min)) {
        jmin = N.y;
      } else {
        jmin = floor(alpha2index(alpha_min, p1.y, p2.y, b.y, d.y));
      }
      if (eq(alpha_max, alphax_max)) {
        jmax = 0;
      } else {
        jmax = ceil(alpha2index(alpha_max, p1.y, p2.y, b.y, d.y));
      }
    } else if (p1.y > p2.y) {
      if (eq(alpha_max, alphay_max)) {
        jmax = N.y;
      } else {
        jmax = floor(alpha2index(alpha_max, p1.y, p2.y, b.y, d.y));
      }
      if (eq(alpha_min, alphay_min)) {
        jmin = 0;
      } else {
        jmin = ceil(alpha2index(alpha_min, p1.y, p2.y, b.y, d.y));
      }
    }

    if (p1.z < p2.z) {
      if (eq(alpha_min, alphaz_min)) {
        kmin = N.z;
      } else {
        kmin = floor(alpha2index(alpha_min, p1.z, p2.z, b.z, d.z));
      }
      if (eq(alpha_max, alphaz_max)) {
        kmax = 0;
      } else {
        kmax = ceil(alpha2index(alpha_max, p1.z, p2.z, b.z, d.z));
      }
    } else if (p1.z > p2.z) {
      if (eq(alpha_max, alphaz_max)) {
        kmax = N.z;
      } else {
        kmax = floor(alpha2index(alpha_max, p1.z, p2.z, b.z, d.z));
      }
      if (eq(alpha_min, alphaz_min)) {
        kmin = 0;
      } else {
        kmin = ceil(alpha2index(alpha_min, p1.z, p2.z, b.z, d.z));
      }
    }
  }
  return MinMaxPlaneIndicesResult{imin, imax,
                                  jmin, jmax,
                                  kmin, kmax,
                                  alpha_min, alpha_max};
}

int plane_index(int imin, int imax) {
    if (imin > imax) return imin - 1;
    else return imin + 1;
}

Point find_first_intersection_alphas(
    Point const& p1, Point const& p2, Point const& b, Point const& d,
    int imin, int imax, int jmin, int jmax, int kmin, int kmax,
    double alpha_min, double alpha_max) {
  Point a{10.0, 10.0, 10.0};

  if (eq(alpha_min, index2alpha(imin, p1.x, p2.x, b.x, d.x))) {
    int index = plane_index(imin, imax);
    a.x = index2alpha(index, p1.x, p2.x, b.x, d.x);
  } else {
    a.x = index2alpha(imin, p1.x, p2.x, b.x, d.x);
  }

  if (eq(alpha_min, index2alpha(jmin, p1.y, p2.y, b.y, d.y))) {
    int index = plane_index(jmin, jmax);
    a.y = index2alpha(index, p1.y, p2.y, b.y, d.y);
  } else {
    a.y = index2alpha(jmin, p1.y, p2.y, b.y, d.y);
  }

  if (eq(alpha_min, index2alpha(kmin, p1.z, p2.z, b.z, d.z))) {
    int index = plane_index(kmin, kmax);
    a.z = index2alpha(index, p1.z, p2.z, b.z, d.z);
  } else {
    a.z = index2alpha(kmin, p1.z, p2.z, b.z, d.z);
  }

  if (a.x < alpha_min || a.x > alpha_max) a.x = 10.0;
  if (a.y < alpha_min || a.y > alpha_max) a.y = 10.0;
  if (a.z < alpha_min || a.z > alpha_max) a.z = 10.0;
  return a;
}

IndexPoint find_voxel_indices(
    double alpha, Point const& p1, Point const& p2,
    Point const& d, Point const& b) {
  int i = floor(alpha2index(alpha, p1.x, p2.x, b.x, d.x));
  int j = floor(alpha2index(alpha, p1.y, p2.y, b.y, d.y));
  int k = floor(alpha2index(alpha, p1.z, p2.z, b.z, d.z));
  return IndexPoint{i, j, k};
}

char dimension_intersecting_plane(Point const& a) {
  if (a.x <= a.y && a.x <= a.z) return 'x';
  if (a.y <= a.x && a.y <= a.z) return 'y';
  return 'z';
}

void slice_range(double zsource, int Nz, double H, double D, double R, double r,
                 int* kmin, int* kmax) {
  double h = (R + r) * H / D;
  double dz = 2.0 * r / Nz;
  double zmin = zsource - h / 2.0;
  double zmax = zsource + h / 2.0;
  *kmin = floor((zmin + r) / dz) - 3;
  *kmax = ceil((zmax + r) / dz) + 3;
  if (*kmin < 0) *kmin = 0;
  if (*kmax > Nz) *kmax = Nz;
}

double get_dist(Point const& p) {
  return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}
double get_dist(Point const& p1, Point const& p2) {
  return get_dist(Point{p1.x - p2.x, p1.y - p2.y, p1.z - p2.z});
}

template<typename T> 
double siddon_xray_transform(
    T obj[NX][NY][MAXZ_SLICE], Point const& p1, Point const& p2,
    double r, double D,
    IndexPoint const& N, Point const& d, Point const& b,
    int pozk, std::function<double(T const&)> vox_to_prj) {
  // source - detector distance
  double sdd = get_dist(p1, p2);
  double prj = 0.0;

  // find entrace and exit plane indices for the x-ray
  MinMaxPlaneIndicesResult result = min_max_plane_indices(N, p1, p2, b, d);
  if (result.imin == -1 && result.imax == -1 &&
      result.jmin == -1 && result.jmax == -1 &&
      result.kmin == -1 && result.kmax == -1) {
    return prj;
  }
  Point alphaf =
      find_first_intersection_alphas(p1, p2, b, d,
                                     result.imin, result.imax,
                                     result.jmin, result.jmax,
                                     result.kmin, result.kmax,
                                     result.alpha_min, result.alpha_max);
  if (alphaf.x == 10.0 && alphaf.y == 10.0 && alphaf.z == 10.0) {
    // alpha can be in range 0 to 1, 10 means there is no intersection
    // of ray and ROI.
    return prj;
  }

  Point inc_a{d.x / abs(p1.x - p2.x),
              d.y / abs(p1.y - p2.y),
              d.z / abs(p1.z - p2.z)};
  char dim = dimension_intersecting_plane(alphaf);
  int inc_i = 0;
  if (p1(dim) > p2(dim)) inc_i = 1;
  else inc_i = -1;

  double a = (alphaf(dim) + result.alpha_min) / 2.0;
  IndexPoint vox_coor = find_voxel_indices(a, p1, p2, d, b);
  // TODO: ovo je ruzno/
  if (vox_coor.x > (N.x - 1) ||
      vox_coor.y > (N.y - 1) ||
      vox_coor.z > (N.z - 1) ||
      vox_coor.x < 0 ||
      vox_coor.y < 0 ||
      vox_coor.z < pozk) {
    return prj;
  }
  double l = (alphaf(dim) - result.alpha_min) * sdd;

  prj += vox_to_prj(obj[vox_coor.x][vox_coor.y][vox_coor.z - pozk]) * l;

  while (true) {
    vox_coor(dim) += inc_i;
    double alpha_min = alphaf(dim);
    alphaf(dim) += inc_a(dim);
    if (vox_coor.x > (N.x - 1) ||
        vox_coor.y > (N.y - 1) ||
        vox_coor.z > (N.z - 1) ||
        vox_coor.x < 0 ||
        vox_coor.y < 0 ||
        vox_coor.z < pozk) {
      // exit while loop when vox_coor address voxel outside ROI
      break;
    } else {
      dim = dimension_intersecting_plane(alphaf);
      if (p1(dim) > p2(dim)) inc_i = 1;
      else inc_i = -1;
      l = (alphaf(dim) - alpha_min) * sdd;
      prj += vox_to_prj(obj[vox_coor.x][vox_coor.y][vox_coor.z - pozk]) * l;
    }
  }
  return prj;
}

Point ray(double D, double alpha, double w,
          Point const& eu, Point const& ev, Point const& ez) {
  return D * cos(alpha) * ev + D * sin(alpha) * eu + w * ez;
}

Point ray_flat_detector(double D, double u, double w,
                        Point const& eu, Point const& ev, Point const& ez) {
  return D * ev + u * eu + w * ez;
}

double obj[NX][NY][NZ];

void read_from_file(string filename, double obj[NX][NY][NZ]) {
  H5File file( filename.c_str(), H5F_ACC_RDONLY );
  //Open dataset for left frames
  DataSet dataset = file.openDataSet("phantom") ;
  DataSpace dataspace = dataset.getSpace();
   
  //get dataset dimensions
  int rank = dataspace.getSimpleExtentNdims();
  hsize_t* dims_out = new hsize_t[rank];
  int ndims = dataspace.getSimpleExtentDims( dims_out, NULL);
 		
  //set reading hyperslabs to whole dataset
  hsize_t* offset = new hsize_t[ndims];	// hyperslab offset in the file
  memset(offset,0,rank * sizeof(hsize_t));
  dataspace.selectHyperslab( H5S_SELECT_SET, dims_out, offset);
  DataSpace memspace( ndims,dims_out); //describe hyperslab in memory space
  memspace.selectHyperslab( H5S_SELECT_SET, dims_out, offset);
  dataset.read( obj, PredType::NATIVE_DOUBLE, memspace, dataspace);
  file.close();
  delete[] offset;
  delete[] dims_out;
}

void write_to_file(string filename, double Df[NP][NR][NC]) {
  H5File file(filename.c_str(), H5F_ACC_TRUNC);
  hsize_t dims[3] = {NP, NR, NC};
  DataSpace dataspace(3, dims);
  DataSet dataset = file.createDataSet( "Df", PredType::NATIVE_DOUBLE, dataspace);
  dataset.write(Df, PredType::NATIVE_DOUBLE);    
  file.close();
}

double obj_slice[NX][NY][MAXZ_SLICE];
Point grad_slice[NX][NY][MAXZ_SLICE];

void siddon_cone_beam_projection(
    string objFileName,
    Point sp[NP], double s[NP],
    double alpha[NC], double w[NR],
    double r, double R, double D, double H,
    IndexPoint const& N, Point const& d, Point const& b,
    double Df[NP][NR][NC], bool dpc) {
  read_from_file(objFileName, obj);
  
  for (int p = 0; p < NP; ++p) {
    printf("p = %d\n", p);
    // detector coordinate system
    Point eu{-sin(s[p]), cos(s[p]), 0.0};
    Point ev{-cos(s[p]), -sin(s[p]), 0.0};
    Point ez{0.0, 0.0, 1.0};

    int Nkmin, Nkmax;
    slice_range(sp[p].z, NZ, H, D, R, r, &Nkmin, &Nkmax);

    for (int x = 0; x < NX; ++x) {
      for (int y = 0; y < NY; ++y) {
        for (int z = Nkmin; z <= Nkmax; ++z) {
          if (!dpc) {
            obj_slice[x][y][z - Nkmin] = obj[x][y][z];
          } else {
            grad_slice[x][y][z - Nkmin] =
                localGradient(obj, IndexPoint{x, y, z}, N, d);
          }
        }
      }
    }
    
    for (int i = 0; i < NR; ++i) {
      for (int j = 0; j < NC; ++j) {
        Point theta = ray(D, alpha[j], w[i], eu, ev, ez);
        Point dp = theta + sp[p];
        if (!dpc) {
          std::function<double(double const&)> vox_to_prj =
              [](double const& x) { return x; };
          Df[p][NR-1-i][j] =
              siddon_xray_transform(obj_slice, dp, sp[p], r, D,
                                    IndexPoint{NX, NY, Nkmax}, d, b,
                                    Nkmin, vox_to_prj);
        } else {
          theta = theta / get_dist(theta);
          Point refv{theta.y, -theta.x, 0.0};
          std::function<double(Point const&)> vox_to_prj =
              [&refv](Point const& x) { return dot(x, refv); };
          Df[p][NR-1-i][j] =
              siddon_xray_transform(grad_slice, dp, sp[p], r, D,
                                    IndexPoint{NX, NY, Nkmax}, d, b,
                                    Nkmin, vox_to_prj);
        }
      }
    }
  }
}

void siddon_cone_beam_projection_flat_detector(
    string objFileName,
    Point sp[NP], double s[NP],
    double u[NC], double w[NR],
    double r, double R, double D, double H,
    IndexPoint const& N, Point const& d, Point const& b,
    double Df[NP][NR][NC], bool dpc) {
  read_from_file(objFileName, obj);
  
  for (int p = 0; p < NP; ++p) {
    printf("p = %d\n", p);
    // detector coordinate system
    Point eu{-sin(s[p]), cos(s[p]), 0.0};
    Point ev{-cos(s[p]), -sin(s[p]), 0.0};
    Point ez{0.0, 0.0, 1.0};

    int Nkmin, Nkmax;
    slice_range(sp[p].z, NZ, H, D, R, r, &Nkmin, &Nkmax);

    for (int x = 0; x < NX; ++x) {
      for (int y = 0; y < NY; ++y) {
        for (int z = Nkmin; z <= Nkmax; ++z) {
          if (!dpc) {
            obj_slice[x][y][z - Nkmin] = obj[x][y][z];
          } else {
            grad_slice[x][y][z - Nkmin] =
                localGradient(obj, IndexPoint{x, y, z}, N, d);
          }
        }
      }
    }
    
    for (int i = 0; i < NR; ++i) {
      for (int j = 0; j < NC; ++j) {
        Point theta = ray_flat_detector(D, u[j], w[i], eu, ev, ez);
        Point dp = theta + sp[p];
        if (!dpc) {
          std::function<double(double const&)> vox_to_prj =
              [](double const& x) { return x; };
          Df[p][NR-1-i][j] =
              siddon_xray_transform(obj_slice, dp, sp[p], r, D,
                                    IndexPoint{NX, NY, Nkmax}, d, b,
                                    Nkmin, vox_to_prj);
        } else {
          theta = theta / get_dist(theta);
          Point refv{theta.y, -theta.x, 0.0};
          std::function<double(Point const&)> vox_to_prj =
              [&refv](Point const& x) { return dot(x, refv); };
          Df[p][NR-1-i][j] =
              siddon_xray_transform(grad_slice, dp, sp[p], r, D,
                                    IndexPoint{NX, NY, Nkmax}, d, b,
                                    Nkmin, vox_to_prj);
        }
      }
    }
  }
}

double Df[NP][NR][NC];

int main() {
  IndexPoint N{NX, NY, NZ};
  Point d{0.001953125,0.001953125,0.001953125};
  Point b{-1.0, -1.0, -1.0};
  double r = 1.0;
  double R = 3.0;
  double D = 2.0 * R;
  double H = 0.5;
  double P = 0.29066822283489996;
  double delta_w = H / NR;
  double delta_alpha = H / NR / D;
  double alpha_shift = 0.0;
  double w_shift = 0.0;
  double alpha[NC];
  linspace(-delta_alpha / 2 * (NC - 1) - alpha_shift,
           delta_alpha / 2 * (NC - 1) - alpha_shift,
           NC, alpha);
  double w[NR];
  linspace(-delta_w / 2 * (NR - 1) - w_shift,
           delta_w / 2 * (NR - 1) - w_shift,
           NR, w);
  double s[NP];
  double ds = 0.5 / 16 /6;
  arange(-8.54567952444, -2.94216803617, NP, ds, s);
  Point sp[NP];
  for (int i = 0; i < NP; ++i) {
    sp[i] = Point{R * cos(s[i]),
                  R * sin(s[i]),
                  P / (2.0 * pi) * s[i]};
  }
  //for (int i = 0; i < NR; ++i) {
  // printf("%.5lf\n", w[i]);
  //}
  siddon_cone_beam_projection("phantom4.h5",
                              sp, s, alpha, w, r, R, D, H, N, d, b, Df,
                              /*dpc=*/false);
  write_to_file("df_1024_2x.h5", Df);
  return 0;
}
