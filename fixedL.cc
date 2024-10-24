#include "itensor/mps/sweeps.h"
#include "itensor/util/input.h"
#include "itensor/util/print_macro.h"
#include "paralleldo.h"
#include "util.h"
#include <future>

#include <algorithm>
#include <chrono>

using namespace itensor;
using std::array;
using std::min;
using std::move;
using std::string;
using std::vector;

static const size_t LABELS_COUNT = 10;

// clang-format off
#define TIME_IT(block, duration)                                     \
    {                                                                \
        auto start_time = std::chrono::high_resolution_clock::now(); \
        block                                                        \
        auto end_time = std::chrono::high_resolution_clock::now();   \
        duration = end_time - start_time;                            \
    }
// clang-format on

// Struct holding info about training "states"
// Could have a more generic name since it can technically store test "states" as well, but probably not necessary (?)
struct TrainingState {
    SiteSet const &sites_;
    const int label = -1;
    const int local_dimension = 0;
    // Not sure what this is supposed to represent, maybe \tilde{\phi}_n in paper (Figure 6(b)) (?)
    // effective image (4 site) tensor, mentioned later in code
    ITensor v;
    vector<Real> data;
    // vector<ITensor> data2;

    template <typename Func, typename ImgType>
    TrainingState(SiteSet const &sites, int label, ImgType const &img, Func const &phi)
        : sites_(sites), label(label), local_dimension(sites(1).m()) {
        auto pixel_count = sites.N();
        data.resize(pixel_count * local_dimension);
        auto i = 0;
        for (auto j : range1(pixel_count)) {
            for (auto n : range1(local_dimension)) {
                data.at(i) = phi(img(j), n);
                ++i;
            }
        }
        // data2 = vector<ITensor>(pixel_count);
        // for (auto j : range1(pixel_count)) {
        //     auto store = DenseReal(local_dimension);
        //     for (auto n : range(local_dimension)) {
        //         store[n] = phi(img(j), n + 1);
        //     }
        //     data2.at(j - 1) = ITensor(IndexSet{sites_(j)}, std::move(store));
        // }
    }

    // Surely it would be better to just store all the data as a vector of ITensor objects (?)
    // A is 1-indexed (?)
    ITensor A(int i) const {
        auto store = DenseReal(local_dimension);
        for (auto n : range(local_dimension)) {
            // TODO: change .at() to []
            store[n] = data.at(local_dimension * (i - 1) + n);
        }
        return ITensor(IndexSet{sites_(i)}, std::move(store));
    }
    // ITensor const &A(int i) const {
    //     return data2.at(i - 1);
    // }
};

class TrainingSet {
  public:
    vector<TrainingState> ts_;
    // Name pixel_count & site_count can be used interchangeably, (?)
    // This is the amount of sites each TrainingState in ts_ has
    int site_count = 0;

    TrainingSet(vector<TrainingState> &&ts, int N_, int thread_count, int batch_count = 1)
        : ts_(move(ts)), site_count(N_), batch_count_(batch_count), thread_count_(thread_count) {
        const int training_image_count = ts_.size();
        batch_length_ = training_image_count / batch_count;
        const int rem = training_image_count % batch_count;
        if (rem != 0) {
            Error(format("training_image_count=%d, batch_count=%d, training_image_count %% batch_count=%d\n"
                         "training_image_count not commensurate with batch_count",
                         training_image_count, batch_count, rem));
        }
        pd_ = ParallelDo(thread_count_, batch_length_);
        for (auto &b : pd_.bounds()) {
            printfln("Thread %d %d -> %d (%d)", b.n, b.begin, b.end, b.size());
        }

        buffer_1 = vector<ITensor>(batch_length_);
        buffer_2 = vector<ITensor>(batch_length_);
    }

    int size() const { return ts_.size(); }

    int thread_count() const { return thread_count_; }

    TrainingState const &front() const { return ts_.front(); }

    static string &writeDir() {
        static string wd = "proj_images";
        return wd;
    }

    void init(MPS const &W) {
        if (not dir_is_made_) {
            auto cmd = "mkdir -p " + writeDir();
            int info = std::system(cmd.c_str());
            if (info != 0) {
                Error(format("Failed execution: \"%s\"}", cmd.c_str()));
            }
            dir_is_made_ = true;
        }
        // This is so ugly but I don't know how to fix it with the way the write system is setup (?)
        auto nextE = vector<ITensor>(batch_length_);
        auto currE = vector<ITensor>(batch_length_);
        for (auto batch_idx : range(batch_count_)) {
            auto batch_start = batch_idx * batch_length_;
            for (auto site_idx = site_count; site_idx >= 3; --site_idx) {
                pd_([&](Bound b) {
                    for (auto i = b.begin; i < b.end; ++i) {
                        auto &t = ts_.at(batch_start + i);
                        nextE.at(i) = t.A(site_idx) * W.A(site_idx);
                        if (site_idx != site_count) {
                            nextE.at(i) *= currE.at(i);
                        }
                        nextE.at(i).scaleTo(1.);
                    }
                });
                currE.swap(nextE);
                writeToFile(fname(batch_idx, site_idx), currE);
            }
        }
        setBond(1);
    }

    void setBond(int b) {
        if (currb_ == b) {
            return;
        }
        currb_ = b;
        // These mean what (?) Maybe indices for the left and right cores of the bond tensor being optimized?
        auto lc = b - 1;
        auto rc = b + 2;
        auto useL = (lc > 0);
        auto useR = (rc < site_count + 1);

        auto &LE = buffer_1;
        auto &RE = buffer_2;
        // Make effective image (4 site) tensors
        // Store in t.v of each elem t of ts
        for (auto batch_idx : range(batch_count_)) {
            auto batch_start = batch_idx * batch_length_;
            if (useL) {
                readFromFile(fname(batch_idx, lc), LE);
            }
            if (useR) {
                readFromFile(fname(batch_idx, rc), RE);
            }
            pd_([&](Bound b) {
                for (auto i = b.begin; i < b.end; ++i) {
                    auto &t = ts_.at(batch_start + i);
                    t.v = t.A(lc + 1) * t.A(rc - 1);
                    if (useL) {
                        t.v *= LE.at(i);
                    }
                    if (useR) {
                        t.v *= RE.at(i);
                    }
                }
            });
        }
    }

    void shiftE(MPS const &W, int b, Direction dir) {
        auto c = (dir == Fromleft) ? b : b + 1;
        auto dc = (dir == Fromleft) ? +1 : -1;

        auto prevc = (dir == Fromleft) ? b - 1 : b + 2;
        auto hasPrev = (prevc >= 1 && prevc <= site_count);

        // if (hasPrev) {
        //     printfln("## Advancing E from %d to %d", prevc, c);
        // } else {
        //     printfln("## Making new E at %d", c);
        // }  // pc

        auto &prevE = buffer_1;
        auto &nextE = buffer_2;
        for (auto batch_idx : range(batch_count_)) {
            auto batch_start = batch_idx * batch_length_;
            if (hasPrev) {
                readFromFile(fname(batch_idx, prevc), prevE);
            }
            pd_([&](Bound b) {
                for (auto i = b.begin; i < b.end; ++i) {
                    auto &t = ts_.at(batch_start + i);
                    nextE.at(i) = t.A(c) * W.A(c);
                    if (hasPrev) {
                        nextE.at(i) *= prevE.at(i);
                    }
                    nextE.at(i).scaleTo(1.);
                }
            });
            writeToFile(fname(batch_idx, c), nextE);
        }
    }

    template <typename Func> void execute(Func &&f) const {
        for (auto batch_idx : range(batch_count_)) {
            auto batch_start = batch_idx * batch_length_;
            pd_([&f, batch_start, this](Bound b) {
                // printfln("B %d %d->%d",b.n,b.begin,b.end);
                for (auto i = batch_start + b.begin; i < batch_start + b.end; ++i) {
                    auto &t = ts_.at(i);
                    f(b.n, t);
                }
            });
        }
    }

  private:
    int currb_ = -1; // left env built to here
    bool dir_is_made_ = false;
    const int batch_count_ = 1;
    // This should be made const as well
    int batch_length_ = 0;
    const int thread_count_ = 1;
    ParallelDo pd_;
    vector<ITensor> buffer_1;
    vector<ITensor> buffer_2;

    string fname(int nb, int j) { return format("%s/B%03dE%05d", writeDir(), nb, j); }

    // ITensor&
    // E(int x, int nt)
    //     {
    //     return E_.at(x).at(nt);
    //     }
    // ITensor const&
    // E(int x, int nt) const
    //     {
    //     return E_.at(x).at(nt);
    //     }
};

//
// Compute squared distance of the actual output
// of the model from the ideal output
//
Real quadcost(ITensor B, TrainingSet const &ts, Args const &args = Args::global()) {
    auto training_image_count = ts.size();
    auto lambda = args.getReal("lambda", 0.);
    auto showlabels = args.getBool("ShowLabels", false);

    auto L = findtype(B, Label);
    if (!L) {
        L = findtype(ts.front().v, Label);
    }
    if (!L) {
        Print(B);
        Print(ts.front().v);
        Error("Couldn't find Label index in quadcost");
    }

    if (args.getBool("Normalize", false)) {
        B /= norm(B);
    }

    //
    // Set up containers for multithreaded calculations
    auto deltas = array<ITensor, LABELS_COUNT>{};
    auto reals = array<vector<Real>, LABELS_COUNT>{};
    for (auto l : range(LABELS_COUNT)) {
        deltas[l] = setElt(L(1 + l));
        reals[l] = vector<Real>(ts.thread_count(), 0.);
    }
    auto ints = vector<int>(ts.thread_count(), 0);
    //

    ts.execute([&](int nt, TrainingState const &t) {
        auto weights = array<Real, LABELS_COUNT>{};
        auto P = B * t.v;
        auto dP = deltas[t.label] - P;
        // sqr instead of something like pow2 for computing the square is wild (?)
        reals[t.label].at(nt) += sqr(norm(dP));
        for (auto l : range(LABELS_COUNT)) {
            weights[l] = std::abs(P.real(L(1 + l)));
        }
        // print(t.n,": "); for(auto w : weights) print(" ",w); println();
        if (t.label == argmax(weights)) {
            ints.at(nt) += 1;
        }
    });

    auto CR = lambda * sqr(norm(B));
    auto C = 0.;
    // This just assumes the labels are in order (?)
    for (auto l : range(LABELS_COUNT)) {
        auto CL = stdx::accumulate(reals[l], 0.);
        if (showlabels) {
            printfln("  Label l=%d C%d = %.10f", l, l, CL / training_image_count);
        }
        C += CL;
    }
    if (showlabels) {
        printfln("  Reg. cost CR = %.10f", CR / training_image_count);
    }
    C += CR;
    // auto ncor = stdx::accumulate(ints, 0);
    // auto ninc = (training_image_count - ncor);
    // printfln("Percent correct = %.4f%%, # incorrect = %d/%d", ncor * 100. / training_image_count, ninc, ncor + ninc);
    // // pc
    return C;
}

//
// Conjugate gradient
//
void cgrad(ITensor &B, TrainingSet &ts, Args const &args) {
    auto training_image_count = ts.size();
    auto Npass = args.getInt("Npass");
    auto lambda = args.getReal("lambda", 0.);
    auto cconv = args.getReal("cconv", 1E-10);
    // printfln("In cgrad, lambda = %.3E", lambda); // pc

    auto L = findtype(B, Label);
    if (!L) {
        L = findtype(ts.front().v, Label);
    }
    if (!L) {
        Error("Couldn't find Label index in cgrad");
    }

    // kronecker deltas, so like y_{n}^{L_n} in section 4 in the paper (?)
    auto deltas = array<ITensor, LABELS_COUNT>{};
    for (auto l : range(LABELS_COUNT)) {
        // "A single element ITensor is an ITensor constructed using the setElt function. It has exactly one non-zero
        // element, which can be any element."
        // I guess this is like a sparse tensor constructor (?)
        deltas[l] = setElt(L(1 + l));
    }

    // Workspace for parallel ops
    auto thread_count = ts.thread_count();
    auto tensors = vector<ITensor>(thread_count, ITensor{});
    auto reals = vector<Real>(thread_count);

    // Compute initial gradient
    ts.execute([&](int nt, TrainingState const &t) {
        // This is the predicted label with the current weight tensor, fig. 6(c) in paper
        auto P = B * t.v;
        // This is the difference between the true label and the predicted label, dark square in fig. 6(d) in paper
        auto dP = deltas[t.label] - P;
        // This is the terms in the sum in gradient, eq. 7 / fig. 6(d) in paper
        // dag is abbreviation for dagger, meaning hermitian conjugation (?)
        tensors.at(nt) += dP * dag(t.v);
    });

    // r is the full gradient, why this name???
    auto r = stdx::accumulate(tensors, ITensor{});
    // regularization penalty to include in the cost function, not mentioned in paper but common ML technique (?)
    if (lambda != 0.) {
        r = r - lambda * B;
    }

    auto p = r;
    for (auto pass : range1(Npass)) {
        // println("  Conj grad pass ", pass); // pc
        // Compute p*A*p
        std::fill(reals.begin(), reals.end(), 0.);
        ts.execute([&](int nt, TrainingState const &t) {
            // The matrix A is like outer
            // product of dag(v) and v, so
            // dag(p)*A*p is |p*v|^2
            auto pv = p * t.v;
            reals.at(nt) += sqr(norm(pv));
        });
        auto pAp = stdx::accumulate(reals, 0.);
        pAp += lambda * sqr(norm(p));

        auto a = sqr(norm(r)) / pAp;
        B = B + a * p;
        B.scaleTo(1.);

        if (pass == Npass) {
            break;
        }

        // Compute new gradient and cost function
        std::fill(tensors.begin(), tensors.end(), ITensor());
        std::fill(reals.begin(), reals.end(), 0.);
        ts.execute([&](int nt, TrainingState const &t) {
            auto P = B * t.v;
            auto dP = deltas[t.label] - P;
            tensors.at(nt) += dP * dag(t.v);
            reals.at(nt) += sqr(norm(dP));
        });
        auto new_r = stdx::accumulate(tensors, ITensor{});
        if (lambda != 0.) {
            new_r = new_r - lambda * B;
        }
        auto beta = sqr(norm(new_r) / norm(r));
        r = new_r;
        r.scaleTo(1.);

        auto C = stdx::accumulate(reals, 0.);
        C += lambda * sqr(norm(B));
        // printfln("  Cost = %.10f", C / training_image_count); // pc

        // Quit if gradient gets too small
        if (norm(r) < cconv) {
            // printfln("  |r| = %.1E < %.1E, breaking", norm(r), cconv); // pc
            break;
            // } else {
            // printfln("  |r| = %.1E", norm(r)); // pc
        }

        p = r + beta * p;
        p.scaleTo(1.);
    }
}

//
// M.L. DMRG
//
void mldmrg(MPS &W, TrainingSet &ts, Sweeps const &sweeps, Args args) {
    auto N = W.N();
    auto training_image_count = ts.size();

    auto method = args.getString("Method");
    auto replace = args.getBool("Replace", false);
    auto pause_step = args.getBool("PauseStep", false);

    auto thread_count = ts.thread_count();
    auto reals = vector<Real>(thread_count);

    auto cargs = Args{args, "Normalize", false};

    int t_idx = 0;
    int timings_count = 2 * (N - 1);
    auto setBond_timings = vector<std::chrono::duration<double, std::milli>>(timings_count);
    auto createB_timings = vector<std::chrono::duration<double, std::milli>>(timings_count);
    auto cgrad_timings = vector<std::chrono::duration<double, std::milli>>(timings_count);
    auto svd_timings = vector<std::chrono::duration<double, std::milli>>(timings_count);
    auto shiftE_timings = vector<std::chrono::duration<double, std::milli>>(timings_count);

    // For loop over sweeps of the MPS
    for (auto sw : range1(sweeps)) {
        printfln("\nSweep %d max_m=%d min_m=%d", sw, sweeps.maxm(sw), sweeps.minm(sw));
        auto svd_args =
            Args{"Cutoff", sweeps.cutoff(sw), "Maxm", sweeps.maxm(sw), "Minm", sweeps.minm(sw), "Sweep", sw};
        // Loop over individual bonds of the MPS
        for (int bond_idx = 1, ha = 1; ha <= 2; sweepnext(bond_idx, ha, N)) {
            // c and c+dc are j,j+1 if sweeping right
            // if sweeping left they are j,j-1
            auto c = (ha == 1) ? bond_idx : bond_idx + 1;
            auto dc = (ha == 1) ? +1 : -1;

            // printfln("\nSweep %d Half %d Bond %d", sw, ha, c);

            // ts.setBond(bond_idx);
            // clang-format off
            TIME_IT(
            ts.setBond(bond_idx);
            , setBond_timings.at(t_idx));
            // clang-format on

            // auto old_m = commonIndex(W.A(c), W.A(c + dc)).m();
            // B is the bond tensor we will optimize
            // auto B = W.A(c) * W.A(c + dc);
            // B.scaleTo(1.);
            ITensor B;
            // clang-format off
            TIME_IT(
            B = W.A(c) * W.A(c + dc); 
            B.scaleTo(1.);
            , createB_timings.at(t_idx));
            // clang-format on

            //
            // Optimize bond tensor B
            //
            // if (method == "conj") {
            //     cgrad(B, ts, args);
            // } else {
            //     Error(format("method type \"%s\" not recognized", method));
            // }
            // clang-format off
            TIME_IT(
            if (method == "conj") {
                cgrad(B, ts, args);
            } else {
                Error(format("method type \"%s\" not recognized", method));
            }
            , cgrad_timings.at(t_idx));
            // clang-format on

            //
            // SVD B back apart into MPS tensors
            //
            // ITensor S;
            // auto spec = svd(B, W.Aref(c), S, W.Aref(c + dc), svd_args);
            // W.Aref(c + dc) *= S;
            // clang-format off
            TIME_IT(
            ITensor S;
            auto spec = svd(B, W.Aref(c), S, W.Aref(c + dc), svd_args);
            W.Aref(c + dc) *= S;
            , svd_timings.at(t_idx));
            // clang-format on
            // auto new_m = commonIndex(W.A(c), W.A(c + dc)).m();
            // printfln("SVD trunc err = %.2E", spec.truncerr()); // pc

            // printfln("Original m=%d, New m=%d", old_m, new_m); // pc

            // auto new_B = W.A(c) * W.A(c + dc);
            // Print(norm(new_B)); // pc
            // printfln("rank(new_B) = %d", rank(new_B)); // pc
            // printfln("|B-new_B| = %.3E", norm(B - new_B)); // pc

            // auto new_quadratic_cost = quadcost(new_B, ts, {cargs, "ShowLabels", true});
            // auto new_quadratic_cost = quadcost(new_B, ts, cargs);
            // printfln("--> After SVD, Cost = %.10f", new_quadratic_cost / training_image_count); // pc

            //
            // Update E's (MPS environment tensors)
            // i.e. projection of training images into current "wings"
            // of the MPS W
            //
            // ts.shiftE(W, bond_idx, ha == 1 ? Fromleft : Fromright);
            // clang-format off
            TIME_IT(
            ts.shiftE(W, bond_idx, ha == 1 ? Fromleft : Fromright);
            , shiftE_timings.at(t_idx));
            // clang-format on

            // if (fileExists("WRITE_WF")) {
            //     println("File WRITE_WF found");
            //     auto cmd = "rm -f WRITE_WF";
            //     int info = system(cmd);
            //     if (info != 0) {
            //         Error(format("Failed execution: \"%s\"}", cmd));
            //     }
            //     println("Writing W to disk");
            //     writeToFile("W", W);
            // }

            // if (fileExists("LAMBDA")) {
            //     auto lf = std::ifstream("LAMBDA");
            //     Real lambda = 0.;
            //     lf >> lambda;
            //     lf.close();
            //     args.add("lambda", lambda);
            //     auto cmd = "rm -f LAMBDA";
            //     int info = system(cmd);
            //     if (info != 0) {
            //         Error(format("Failed execution: \"%s\"}", cmd));
            //     }
            //     println("new lambda = ", lambda);
            // }

            if (pause_step) {
                PAUSE;
            }
            t_idx++;

        } // loop over c,dc

        println("Writing W to disk");
        writeToFile("W", W);

        // These are temporarily here
        auto new_B = W.A(1) * W.A(2);
        auto new_quadratic_cost = quadcost(new_B, ts, cargs);
        printfln("--> After Sweep, Cost = %.10f", new_quadratic_cost / training_image_count);

        std::vector<std::reference_wrapper<std::vector<std::chrono::duration<double, std::milli>>>> all_timings = {
            std::ref(setBond_timings), std::ref(createB_timings), std::ref(cgrad_timings), std::ref(svd_timings),
            std::ref(shiftE_timings)};

        std::cout << "Time for each bond [ms]\n";
        std::cout << "t_idx setBond createB cgrad svd shiftE\n";
        for (int i : range(t_idx)) {
            std::cout << i << " ";
            for (auto &t : all_timings) {
                std::cout << t.get().at(i).count() << " ";
            }
            std::cout << "\n";
        }

        std::cout << "Average time for each bond [ms]\n";
        std::cout << "setBond createB cgrad svd shiftE\n";
        auto all_avg_timings = vector<double>(all_timings.size());
        for (auto i : range(all_timings.size())) {
            all_avg_timings.at(i) =
                (stdx::accumulate(all_timings.at(i).get(), std::chrono::duration<double, std::milli>(0)).count() /
                 timings_count);
            std::cout << all_avg_timings.at(i) << " ";
        }
        std::cout << "\n";

        std::cout << "Average Timings for each bond [%]\n";
        std::cout << "setBond createB cgrad svd shiftE\n";
        auto total_avg_time = stdx::accumulate(all_avg_timings, 0.);
        for (auto &t : all_avg_timings) {
            std::cout << t / total_avg_time << " ";
        }
        std::cout << "\n";

    } // loop over sweeps

} // mldmrg

int main(int argc, const char *argv[]) {
    // Set environment variables to use 1 thread
    setOneThread();

    if (argc != 2) {
        printfln("Usage: %s inputfile", argv[0]);
        return 0;
    }
    auto input = InputGroup(argv[1], "input");

    int local_dimension = 2;
    auto data_dir = input.getString("datadir", "/Users/mstoudenmire/software/tnml/mllib/MNIST");
    auto max_training_image_count_per_label = input.getInt("Ntrain", 60000);
    auto batch_count = input.getInt("Nbatch", 10);
    auto sweep_count = input.getInt("Nsweep", 50);
    auto cutoff = input.getReal("cutoff", 1E-10);
    auto maxm = input.getInt("maxm", 5000);
    auto minm = input.getInt("minm", max(10, maxm / 2));
    auto ninitial = input.getInt("ninitial", 100);
    auto thread_count = input.getInt("nthread", 1);
    auto replace = input.getYesNo("replace", false);
    auto pause_step = input.getYesNo("pause_step", false);
    // auto feature = input.getString("feature","normal");

    // Cost function settings
    auto lambda = input.getReal("lambda", 0.);

    // Gradient settings
    auto method = input.getString("method", "conj");
    auto alpha = input.getReal("alpha", 0.01);
    auto clip = input.getReal("clip", 1.0);
    auto Npass = input.getInt("Npass", 4);
    auto cconv = input.getReal("cconv", 1E-10);

    auto labels = array<long, LABELS_COUNT>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};

    auto training_images = readMNIST(data_dir, mllib::Train, {"NT=", max_training_image_count_per_label});
    auto pixels_per_image = training_images.front().size();
    // Is this just where the labels index will be?
    auto c = pixels_per_image / 2;
    printfln("%d sites of dimension %d", pixels_per_image, local_dimension);
    SiteSet sites;
    auto sites_filename = "sites";
    if (fileExists(sites_filename)) {
        sites = readFromFile<SiteSet>(sites_filename);
        if (sites(1).m() != (long)local_dimension) {
            printfln("Error: d=%d but dimension of first site is %d", local_dimension, sites(1).m());
            EXIT
        }
    } else {
        sites = SiteSet(pixels_per_image, local_dimension);
        writeToFile(sites_filename, sites);
    }

    //
    // Local feature map (a lambda function)
    //
    auto phi = [](Real g, int n) -> Real {
        if (g < 0 || g > 255.) {
            Error(format("Expected g=%f to be in [0,255]", g));
        }
        auto x = g / 255.;
        return pow(x / 4., n - 1);
    };

    println("Converting training set to MPS");
    auto states = vector<TrainingState>();
    for (auto &img : training_images) {
        states.emplace_back(sites, img.label, img, phi);
    }

    int training_image_count = states.size();
    printfln("Total of %d training images", training_image_count);

    auto ts = TrainingSet(move(states), pixels_per_image, thread_count, batch_count);

    Index L;
    MPS W;
    auto W_init_file = "Wstart";
    if (fileExists(W_init_file)) {
        printfln("Reading W from \"%s\"", W_init_file);
        W = readFromFile<MPS>(W_init_file, sites);
        L = findtype(W.A(c), Label);
        if (!L) {
            printfln("Expected W to have Label type Index at site %d", c);
            EXIT
        }
    } else if (fileExists("W0")) {
        println("Found separate W0,W1,...,W9 MPS: summing");
        L = Index("L", 10, Label);
        auto Lval = [&L](long n) { return L(1 + n); };
        auto ipsis = vector<MPS>(labels.size());
        for (auto n : range(labels)) {
            auto &in = ipsis.at(n);
            in = readFromFile<MPS>(format("W%d", n));
            // in.position(1);
            in.Aref(c) *= setElt(Lval(labels[n]));
            // PrintData(in.A(c));
        }
        printfln("Summing all %d label states together", ipsis.size());
        W = sum(ipsis, {"Cutoff", 1E-10});
        Print(W.A(c));
        println("Done making initial W");
        writeToFile("W", W);
    } else {
        //
        // If W not read from disk,
        // make initial W by summing training
        // states together
        //
        L = Index("L", 10, Label);
        auto Lval = [&L](long n) { return L(1 + n); };
        auto ipsis = vector<MPS>(labels.size());
        for (auto n : range(labels)) {
            auto psis = vector<MPS>(ninitial);
            for (auto m : range(ninitial)) {
                psis.at(m) = makeMPS(sites, randImg(training_images, labels[n]), phi);
            }
            printfln("Summing %d random label %d states", ninitial, labels[n]);
            ipsis.at(n) = sum(psis, {"Cutoff", 1E-10, "Maxm", 10});
            ipsis.at(n).Aref(c) *= 0.1 * setElt(Lval(labels[n]));
        }
        printfln("Summing all %d label states together", ipsis.size());
        W = sum(ipsis, {"Cutoff", 1E-8, "Maxm", 10});
        W.Aref(c) /= norm(W.A(c));
        println("Done making initial W");
        writeToFile("W", W);
    }
    Print(overlap(W, W));
    println("Done making initial W");

    training_images.clear(); // to save memory

    if (!findtype(W.A(c), Label)) {
        Error(format("Label Index not on site %d", c));
    }

    //
    // Project training states (product states)
    // into environment of W MPS
    //
    print("Projecting training states...");
    ts.init(W);
    println("done");

    println("Calling quadcost...");
    auto C = quadcost(W.A(1) * W.A(2), ts, {"lambda", lambda});
    printfln("Before starting DMRG Cost = %.10f", C / training_image_count);
    if (pause_step) {
        PAUSE;
    }

    auto sweeps = Sweeps(sweep_count, minm, maxm, cutoff);

    auto args = Args{"lambda", lambda, "Method", method, "Npass",   Npass,   "alpha",     alpha,
                     "clip",   clip,   "cconv",  cconv,  "Replace", replace, "PauseStep", pause_step};

    auto start = std::chrono::high_resolution_clock::now();

    mldmrg(W, ts, sweeps, args);

    auto end = std::chrono::high_resolution_clock::now();

    // std::chrono::duration<double, std::milli> duration = end - start;
    // std::cout << "Duration: " << duration.count() << " ms" << std::endl;

    std::chrono::duration<double> duration = end - start;
    std::cout << "Duration: " << duration.count() << " seconds" << std::endl;

    println("Writing W to disk");
    writeToFile("W", W);

    return 0;
}
