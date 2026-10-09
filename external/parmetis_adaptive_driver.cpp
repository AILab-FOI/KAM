#include <mpi.h>
#include <parmetis.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr const char* INPUT_MAGIC = "KAM_BENCH_PARTITION_V1";
constexpr const char* OUTPUT_MAGIC = "KAM_BENCH_PARTITION_RESULT_V1";

struct Problem {
    idx_t n = 0;
    idx_t nparts = 0;
    real_t imbalance_tolerance = static_cast<real_t>(1.05);
    real_t ipc2redist = static_cast<real_t>(1.0);
    std::vector<real_t> capacity_shares;
    std::vector<idx_t> current_part;
    std::vector<idx_t> vertex_weight;
    std::vector<idx_t> migration_size;
    std::vector<std::tuple<idx_t, idx_t, idx_t>> edges;
};

void expect_key(std::istream& in, const std::string& expected) {
    std::string key;
    if (!(in >> key) || key != expected) {
        throw std::runtime_error("expected key '" + expected + "'");
    }
}

Problem read_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input file: " + path);
    std::string magic;
    in >> magic;
    if (magic != INPUT_MAGIC) throw std::runtime_error("invalid input protocol magic");

    Problem p;
    expect_key(in, "n"); in >> p.n;
    expect_key(in, "nparts"); in >> p.nparts;
    expect_key(in, "imbalance_tolerance"); in >> p.imbalance_tolerance;
    expect_key(in, "ipc2redist"); in >> p.ipc2redist;

    if (p.n <= 0 || p.nparts <= 1 || p.nparts > p.n) {
        throw std::runtime_error("invalid n/nparts");
    }

    expect_key(in, "capacity_shares");
    p.capacity_shares.resize(static_cast<std::size_t>(p.nparts));
    for (auto& x : p.capacity_shares) in >> x;

    expect_key(in, "current_part");
    p.current_part.resize(static_cast<std::size_t>(p.n));
    for (auto& x : p.current_part) in >> x;

    expect_key(in, "vertex_weight");
    p.vertex_weight.resize(static_cast<std::size_t>(p.n));
    for (auto& x : p.vertex_weight) in >> x;

    expect_key(in, "migration_size");
    p.migration_size.resize(static_cast<std::size_t>(p.n));
    for (auto& x : p.migration_size) in >> x;

    std::size_t m = 0;
    expect_key(in, "edges"); in >> m;
    p.edges.reserve(m);
    for (std::size_t e = 0; e < m; ++e) {
        idx_t u, v, w;
        in >> u >> v >> w;
        if (u < 0 || v < 0 || u >= p.n || v >= p.n || u == v || w <= 0) {
            throw std::runtime_error("invalid edge in input");
        }
        p.edges.emplace_back(u, v, w);
    }
    if (!in) throw std::runtime_error("truncated/invalid input file");
    return p;
}

void write_result(const std::string& path, const std::vector<idx_t>& part, idx_t edgecut) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open output file: " + path);
    out << OUTPUT_MAGIC << "\n";
    out << "n " << part.size() << "\n";
    out << "edgecut " << edgecut << "\n";
    out << "part";
    for (idx_t x : part) out << ' ' << x;
    out << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int exit_code = 0;
    try {
        std::string input_path, output_path;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
            else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        }
        if (input_path.empty() || output_path.empty()) {
            throw std::runtime_error("usage: parmetis_adaptive_driver --input FILE --output FILE");
        }

        // For benchmark sizes the input is intentionally replicated. The
        // partitioning computation itself remains ParMETIS distributed-memory.
        Problem p = read_problem(input_path);
        if (size > p.n) {
            throw std::runtime_error("MPI rank count must not exceed graph vertex count");
        }

        std::vector<idx_t> vtxdist(static_cast<std::size_t>(size + 1));
        for (int r = 0; r <= size; ++r) {
            vtxdist[static_cast<std::size_t>(r)] =
                static_cast<idx_t>((static_cast<long long>(p.n) * r) / size);
        }
        idx_t first = vtxdist[static_cast<std::size_t>(rank)];
        idx_t last = vtxdist[static_cast<std::size_t>(rank + 1)];
        idx_t local_n = last - first;

        std::vector<std::vector<std::pair<idx_t, idx_t>>> nbrs(static_cast<std::size_t>(local_n));
        for (const auto& e : p.edges) {
            idx_t u, v, w;
            std::tie(u, v, w) = e;
            if (u >= first && u < last) nbrs[static_cast<std::size_t>(u - first)].push_back({v, w});
            if (v >= first && v < last) nbrs[static_cast<std::size_t>(v - first)].push_back({u, w});
        }

        std::vector<idx_t> xadj(static_cast<std::size_t>(local_n + 1), 0);
        for (idx_t i = 0; i < local_n; ++i) {
            xadj[static_cast<std::size_t>(i + 1)] =
                xadj[static_cast<std::size_t>(i)] +
                static_cast<idx_t>(nbrs[static_cast<std::size_t>(i)].size());
        }
        std::vector<idx_t> adjncy(static_cast<std::size_t>(xadj.back()));
        std::vector<idx_t> adjwgt(static_cast<std::size_t>(xadj.back()));
        for (idx_t i = 0; i < local_n; ++i) {
            idx_t pos = xadj[static_cast<std::size_t>(i)];
            for (const auto& nw : nbrs[static_cast<std::size_t>(i)]) {
                adjncy[static_cast<std::size_t>(pos)] = nw.first;
                adjwgt[static_cast<std::size_t>(pos)] = nw.second;
                ++pos;
            }
        }

        std::vector<idx_t> vwgt(static_cast<std::size_t>(local_n));
        std::vector<idx_t> vsize(static_cast<std::size_t>(local_n));
        std::vector<idx_t> part(static_cast<std::size_t>(local_n));
        for (idx_t i = 0; i < local_n; ++i) {
            idx_t g = first + i;
            vwgt[static_cast<std::size_t>(i)] = p.vertex_weight[static_cast<std::size_t>(g)];
            vsize[static_cast<std::size_t>(i)] = p.migration_size[static_cast<std::size_t>(g)];
            part[static_cast<std::size_t>(i)] = p.current_part[static_cast<std::size_t>(g)];
        }

        idx_t wgtflag = 3;  // vertex and edge weights
        idx_t numflag = 0;  // C/zero-based numbering
        idx_t ncon = 1;
        idx_t nparts = p.nparts;
        std::vector<real_t> tpwgts = p.capacity_shares;
        std::vector<real_t> ubvec(1, p.imbalance_tolerance);
        real_t ipc2redist = p.ipc2redist;
        idx_t options[3] = {0, 0, 0};  // ParMETIS defaults
        idx_t edgecut = 0;
        MPI_Comm comm = MPI_COMM_WORLD;

        int status = ParMETIS_V3_AdaptiveRepart(
            vtxdist.data(), xadj.data(),
            adjncy.empty() ? nullptr : adjncy.data(),
            vwgt.data(), vsize.data(),
            adjwgt.empty() ? nullptr : adjwgt.data(),
            &wgtflag, &numflag, &ncon, &nparts,
            tpwgts.data(), ubvec.data(), &ipc2redist,
            options, &edgecut, part.data(), &comm);

        if (status != METIS_OK) {
            throw std::runtime_error("ParMETIS_V3_AdaptiveRepart returned status " + std::to_string(status));
        }

        std::vector<int> recvcounts(static_cast<std::size_t>(size));
        std::vector<int> displs(static_cast<std::size_t>(size));
        for (int r = 0; r < size; ++r) {
            recvcounts[static_cast<std::size_t>(r)] =
                static_cast<int>(vtxdist[static_cast<std::size_t>(r + 1)] - vtxdist[static_cast<std::size_t>(r)]);
            displs[static_cast<std::size_t>(r)] = static_cast<int>(vtxdist[static_cast<std::size_t>(r)]);
        }
        std::vector<idx_t> global_part(rank == 0 ? static_cast<std::size_t>(p.n) : 0);
        MPI_Gatherv(part.data(), static_cast<int>(local_n), IDX_T,
                    rank == 0 ? global_part.data() : nullptr,
                    recvcounts.data(), displs.data(), IDX_T,
                    0, MPI_COMM_WORLD);

        if (rank == 0) write_result(output_path, global_part, edgecut);
    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << "parmetis_adaptive_driver: " << e.what() << std::endl;
        exit_code = 2;
    }

    // Make all ranks return failure if any rank encountered an exception.
    int global_exit = 0;
    MPI_Allreduce(&exit_code, &global_exit, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_exit;
}
