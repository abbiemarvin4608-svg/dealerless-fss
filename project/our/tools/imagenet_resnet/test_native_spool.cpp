#include "cnn_native_spool.h"
#include <iostream>

int main() {
    try {
        int first, second;
        std::vector<uint64_t> a{0, 0xffffffffffffffffULL, 13}, b{7};
        std::vector<uint8_t> choices{1, 0, 1};
        cnn_spool::begin(4096);
        cnn_spool::save(&first, 2, a, choices);
        cnn_spool::save(&second, 3, b);
        if (!a.empty() || a.capacity() || !choices.empty() || b.capacity())
            throw std::runtime_error("prepared payload remains resident");
        cnn_spool::seal();
        cnn_spool::load(&second, 3, b);
        cnn_spool::load(&first, 2, a, choices);
        if (a != std::vector<uint64_t>{0, 0xffffffffffffffffULL, 13} ||
            choices != std::vector<uint8_t>{1, 0, 1} || b != std::vector<uint64_t>{7})
            throw std::runtime_error("native material payload changed");
        bool exhausted = false;
        try { cnn_spool::load(&first, 2, a, choices); }
        catch (const std::runtime_error&) { exhausted = true; }
        if (!exhausted || cnn_spool::loaded != 2) throw std::runtime_error("native material reused");
        cnn_spool::release(a); cnn_spool::release(choices); cnn_spool::release(b);
        if (a.capacity() || choices.capacity() || b.capacity())
            throw std::runtime_error("consumed COT payload remains resident");
        cnn_spool::finish();
        cnn_spool::begin(1);
        a.push_back(13);
        bool limited = false;
        try { cnn_spool::save(&first, 2, a); }
        catch (const std::runtime_error&) { limited = true; }
        cnn_spool::discard();
        if (!limited) throw std::runtime_error("native spool budget ignored");
        std::cout << "native spool exact payload, single consumption, released COT capacity, budget: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        cnn_spool::discard();
        std::cerr << error.what() << '\n';
        return 1;
    }
}
