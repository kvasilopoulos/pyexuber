#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <carma>
#include <armadillo>

#include "exubercore/radf.hpp"

namespace py = pybind11;

namespace {

// Thin translation layer, mirrors exuber's Rcpp wrapper (src/rls_gsadf.cpp):
// unpack the host-language array, call exubercore, repack the result.
// Copies at the boundary rather than relying on CARMA's zero-copy default --
// negligible next to the Monte Carlo/bootstrap work this feeds, and avoids
// ownership bugs. GIL released around the actual computation.
py::array_t<double> radf_stat(py::array_t<double> yxmat, int min_win, int lag) {
  arma::mat m = carma::arr_to_mat<double>(yxmat, /*copy=*/true);

  arma::vec result;
  {
    py::gil_scoped_release release;
    result = exubercore::radf(m, min_win, lag);
  }
  return carma::col_to_arr<double>(result, /*copy=*/true);
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
