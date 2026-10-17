#include "ml_scratch/rnn.hpp"

namespace ml_scratch {

const char* recurrent_cell_name(const RecurrentCell cell) {
    switch (cell) {
    case RecurrentCell::elman:
        return "elman";
    case RecurrentCell::lstm:
        return "lstm";
    case RecurrentCell::gru:
        return "gru";
    }
    return "";
}

} // namespace ml_scratch
