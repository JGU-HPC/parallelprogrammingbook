#ifndef BINARY_IO_HPP
#define BINARY_IO_HPP

#include <string>
#include <fstream>

template <
    typename index_t, 
    typename value_t>
void dump_binary(
    const value_t * data, 
    const index_t length, 
    std::string filename) {

    std::ofstream ofile(filename.c_str(), std::ios::binary);
    ofile.write((char*) data, sizeof(value_t)*length);
    ofile.close();
}

template <
    typename index_t, 
    typename value_t>
void load_binary(
    const value_t * data, 
    const index_t length, 
    std::string filename) {

    std::ifstream ifile(filename.c_str(), std::ios::binary);
    ifile.read((char*) data, sizeof(value_t)*length);
    ifile.close();
}

#endif
