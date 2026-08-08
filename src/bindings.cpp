#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <armadillo>
#include <cstring>

#include "exubercore/radf.hpp"

namespace py = pybind11;

namespace {

// Thin translation layer, mirrors exuber's Rcpp wrapper (src/rls_gsadf.cpp):
// unpack the host-language array, call exubercore, repack the result.
//
// Hand-rolled instead of using CARMA: CARMA unconditionally fetches its own
// pinned Armadillo checkout for its carma::carma target, independent of
// whatever Armadillo exubercore's CMakeLists resolves via find_package --
// two different Armadillo header trees compiled into the same extension
// module is an ODR violation (arma::Mat has two different shapes) that
// segfaults on first call. Since we always copy at the boundary anyway (no
// zero-copy needed), a plain buffer-protocol copy avoids the whole problem:
// there's exactly one Armadillo in this build, the one exubercore uses.
arma::mat to_arma_mat(const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
  py::buffer_info buf = arr.request();
  if (buf.ndim != 2) {
    throw std::invalid_argument("yxmat must be 2-D");
  }
  auto rows = static_cast<arma::uword>(buf.shape[0]);
  auto cols = static_cast<arma::uword>(buf.shape[1]);
  arma::mat m(rows, cols);
  const double* src = static_cast<const double*>(buf.ptr);
  for (arma::uword i = 0; i < rows; ++i) {
    for (arma::uword j = 0; j < cols; ++j) {
      m(i, j) = src[i * cols + j];
    }
  }
  return m;
}

py::array_t<double> to_numpy(const arma::vec& v) {
  py::array_t<double> out(v.n_elem);
  std::memcpy(out.request().ptr, v.memptr(), v.n_elem * sizeof(double));
  return out;
}

py::array_t<double> radf_stat(py::array_t<double, py::array::c_style | py::array::forcecast> yxmat,
                               int min_win, int lag) {
  arma::mat m = to_arma_mat(yxmat);

  arma::vec result;
  {
    py::gil_scoped_release release;
    result = exubercore::radf(m, min_win, lag);
  }
  return to_numpy(result);
}

} // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "pyexuber native extension: thin pybind11 bindings over exubercore";
  m.def("radf_stat", &radf_stat, py::arg("yxmat"), py::arg("min_win"), py::arg("lag") = 0,
        "Recursive least-squares ADF/SADF/GSADF/BSADF statistic vector.\n\n"
        "yxmat: column 0 is the dependent variable (levels), remaining\n"
        "columns are regressors, as produced by pyexuber._unroot(). Throws\n"
        "ValueError (from exubercore's std::invalid_argument) on a bad\n"
        "min_win/lag/shape combination.");
}
