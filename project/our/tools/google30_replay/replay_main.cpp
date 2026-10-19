#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "datatypes.h"
#include "predictors.h"

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  std::ifstream rows(argv[2]);
  std::set<int> selected;
  int index;
  while (rows >> index) selected.insert(index);
  std::ifstream input(argv[1]);
  if (!input) throw std::runtime_error("cannot open input");
  std::string line;
  int row = 0;
  while (std::getline(input, line)) {
    if (!selected.count(row++)) continue;
    std::stringstream parser(line);
    std::string cell;
    std::vector<MYINT> values;
    while (std::getline(parser, cell, ',')) {
      const double scaled = std::ldexp(std::stod(cell), -scaleForX);
      if (!std::isfinite(scaled) || scaled < -32768 || scaled >= 32768)
        throw std::runtime_error("input outside historical int16 domain");
      values.push_back(static_cast<MYINT>(scaled));
    }
    if (values.size() != 3168) throw std::runtime_error("expected 99 x 32 features");
    std::vector<MYINT *> pointers;
    for (auto &value : values) pointers.push_back(&value);
    int32_t prediction;
    seedotFixed(pointers.data(), &prediction);
    std::cout << row - 1 << ',' << prediction << '\n';
  }
}
