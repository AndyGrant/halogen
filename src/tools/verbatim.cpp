// Load the network file, do some preprocessing, and then save it to the build directory for inclusion in the final
// binary

#include "network/arch.hpp"
#include "network/inputs/king_bucket.h"
#include "network/inputs/threat.h"
#include "network/simd/define.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <ranges>
#include <tuple>
#include <utility>

namespace NN
{

struct raw_network
{
    alignas(64) std::array<std::array<int16_t, FT_SIZE>, KingBucket::TOTAL_KING_BUCKET_INPUTS> ft_weight = {};
    alignas(64) std::array<std::array<int8_t, FT_SIZE>, Threats::TOTAL_THREAT_FEATURES> ft_threat_weight = {};
    alignas(64) std::array<int16_t, FT_SIZE> ft_bias = {};
    alignas(64) std::array<std::array<std::array<int16_t, FT_SIZE>, L1_SIZE>, OUTPUT_BUCKETS> l1_weight = {};
    alignas(64) std::array<std::array<int16_t, L1_SIZE>, OUTPUT_BUCKETS> l1_bias = {};
    alignas(64) std::array<std::array<std::array<float, L1_SIZE * 2>, L2_SIZE>, OUTPUT_BUCKETS> l2_weight = {};
    alignas(64) std::array<std::array<float, L2_SIZE>, OUTPUT_BUCKETS> l2_bias = {};
    alignas(64) std::array<std::array<float, L2_SIZE>, OUTPUT_BUCKETS> l3_weight = {};
    alignas(64) std::array<float, OUTPUT_BUCKETS> l3_bias = {};
};

auto shuffle_ft_neurons(const decltype(raw_network::ft_weight)& ft_weight,
    const decltype(raw_network::ft_threat_weight)& ft_threat_weight, const decltype(raw_network::ft_bias)& ft_bias,
    const decltype(raw_network::l1_weight)& l1_weight)
{
    auto ft_weight_output = std::make_unique<decltype(raw_network::ft_weight)>();
    auto ft_threat_weight_output = std::make_unique<decltype(raw_network::ft_threat_weight)>();
    auto ft_bias_output = std::make_unique<decltype(raw_network::ft_bias)>();
    auto l1_weight_output = std::make_unique<decltype(raw_network::l1_weight)>();

#ifdef NETWORK_SHUFFLE
    *ft_weight_output = ft_weight;
    *ft_threat_weight_output = ft_threat_weight;
    *ft_bias_output = ft_bias;
    *l1_weight_output = l1_weight;
#else
    // clang-format off
    constexpr std::array<size_t, FT_SIZE / 2> shuffle_order = {
	   224, 163, 199, 273, 304, 299, 292, 142, 33, 81, 145, 11, 240, 201, 173, 47, 276, 209, 309, 195, 265, 39, 231, 311, 146, 65, 113, 102, 214, 313, 103, 15, 157, 172, 234, 317, 12, 256, 180, 251, 262, 108, 124, 5, 223, 58, 269, 287, 143, 300, 168, 280, 126, 91, 303, 128, 243, 186, 250, 40, 314, 264, 294, 282, 35, 196, 87, 42, 101, 75, 139, 193, 302, 203, 305, 212, 306, 257, 30, 34, 59, 85, 77, 169, 20, 164, 298, 129, 56, 28, 312, 95, 71, 133, 0, 248, 7, 184, 96, 22, 17, 268, 283, 235, 217, 13, 161, 297, 121, 301, 171, 118, 286, 228, 158, 194, 10, 32, 130, 88, 132, 316, 106, 127, 6, 107, 38, 136, 79, 86, 46, 98, 74, 93, 123, 70, 263, 216, 229, 258, 210, 66, 3, 138, 90, 290, 110, 16, 54, 188, 116, 147, 252, 213, 8, 247, 111, 156, 153, 135, 92, 23, 31, 160, 63, 109, 233, 57, 131, 190, 197, 310, 260, 255, 84, 166, 112, 141, 26, 117, 215, 182, 183, 155, 244, 44, 289, 25, 192, 189, 177, 9, 36, 18, 315, 318, 296, 176, 187, 211, 51, 37, 270, 29, 284, 104, 159, 67, 170, 241, 219, 1, 73, 114, 288, 205, 43, 277, 137, 185, 105, 89, 134, 218, 202, 293, 236, 4, 60, 50, 78, 221, 125, 308, 62, 200, 242, 162, 100, 239, 140, 279, 272, 207, 76, 61, 14, 191, 2, 152, 72, 261, 165, 198, 254, 150, 232, 222, 80, 259, 227, 149, 295, 24, 246, 27, 266, 120, 253, 225, 237, 148, 178, 181, 274, 41, 69, 21, 278, 206, 97, 249, 175, 144, 267, 174, 285, 119, 238, 99, 64, 45, 230, 208, 167, 82, 245, 83, 94, 151, 115, 271, 48, 226, 154, 281, 275, 307, 122, 52, 291, 220, 49, 55, 204, 19, 68, 53, 179, 319, };
    // clang-format on

    constexpr auto is_valid_permutation = [](auto order) constexpr
    {
        std::ranges::sort(order);
        return std::ranges::equal(order, std::views::iota(size_t { 0 }, FT_SIZE / 2));
    };
    static_assert(is_valid_permutation(shuffle_order), "shuffle_order must be a valid permutation of [0, FT_SIZE / 2)");

    for (size_t i = 0; i < FT_SIZE / 2; i++)
    {
        auto old_idx = shuffle_order[i];

        (*ft_bias_output)[i] = ft_bias[old_idx];
        (*ft_bias_output)[i + FT_SIZE / 2] = ft_bias[old_idx + FT_SIZE / 2];

        for (size_t j = 0; j < KingBucket::TOTAL_KING_BUCKET_INPUTS; j++)
        {
            (*ft_weight_output)[j][i] = ft_weight[j][old_idx];
            (*ft_weight_output)[j][i + FT_SIZE / 2] = ft_weight[j][old_idx + FT_SIZE / 2];
        }

        for (size_t j = 0; j < Threats::TOTAL_THREAT_FEATURES; j++)
        {
            (*ft_threat_weight_output)[j][i] = ft_threat_weight[j][old_idx];
            (*ft_threat_weight_output)[j][i + FT_SIZE / 2] = ft_threat_weight[j][old_idx + FT_SIZE / 2];
        }

        for (size_t j = 0; j < OUTPUT_BUCKETS; j++)
        {
            for (size_t k = 0; k < L1_SIZE; k++)
            {
                (*l1_weight_output)[j][k][i] = l1_weight[j][k][old_idx];
                (*l1_weight_output)[j][k][i + FT_SIZE / 2] = l1_weight[j][k][old_idx + FT_SIZE / 2];
            }
        }
    }
#endif
    return std::make_tuple(std::move(ft_weight_output), std::move(ft_threat_weight_output), std::move(ft_bias_output),
        std::move(l1_weight_output));
}

auto adjust_for_packus(const decltype(raw_network::ft_weight)& ft_weight,
    const decltype(raw_network::ft_threat_weight)& ft_threat_weight, const decltype(raw_network::ft_bias)& ft_bias)
{
    auto permuted_ft_weight = std::make_unique<decltype(raw_network::ft_weight)>();
    auto permuted_threat_weight = std::make_unique<decltype(raw_network::ft_threat_weight)>();
    auto permuted_bias = std::make_unique<decltype(raw_network::ft_bias)>();

#if defined(USE_AVX2)
#if defined(USE_AVX512)
    constexpr std::array<size_t, 8> mapping = { 0, 4, 1, 5, 2, 6, 3, 7 };
#else
    constexpr std::array<size_t, 4> mapping = { 0, 2, 1, 3 };
#endif

    // Permute FT weight
    for (size_t i = 0; i < ft_weight.size(); i++)
    {
        for (size_t j = 0; j < FT_SIZE; j += SIMD::vec_size)
        {
            for (size_t x = 0; x < SIMD::vec_size; x++)
            {
                size_t target_idx = j + mapping[x / 8] * 8 + x % 8;
                (*permuted_ft_weight)[i][target_idx] = ft_weight[i][j + x];
            }
        }
    }

    // Permute threat weight
    for (size_t i = 0; i < ft_threat_weight.size(); i++)
    {
        for (size_t j = 0; j < FT_SIZE; j += SIMD::vec_size)
        {
            for (size_t x = 0; x < SIMD::vec_size; x++)
            {
                size_t target_idx = j + mapping[x / 8] * 8 + x % 8;
                (*permuted_threat_weight)[i][target_idx] = ft_threat_weight[i][j + x];
            }
        }
    }

    // Permute bias
    for (size_t j = 0; j < FT_SIZE; j += SIMD::vec_size)
    {
        for (size_t x = 0; x < SIMD::vec_size; x++)
        {
            size_t target_idx = j + mapping[x / 8] * 8 + x % 8;
            (*permuted_bias)[target_idx] = ft_bias[j + x];
        }
    }

#else
    (*permuted_ft_weight) = ft_weight;
    (*permuted_threat_weight) = ft_threat_weight;
    (*permuted_bias) = ft_bias;
#endif

    return std::make_tuple(std::move(permuted_ft_weight), std::move(permuted_threat_weight), std::move(permuted_bias));
}

auto rescale_l1_bias(const decltype(raw_network::l1_bias)& input)
{
    auto output = std::make_unique<decltype(raw_network::l1_bias)>();

    // rescale l1 bias to account for FT_activation adjusting to 0-127 scale
    for (size_t i = 0; i < OUTPUT_BUCKETS; i++)
    {
        for (size_t j = 0; j < L1_SIZE; j++)
        {
            (*output)[i][j] = input[i][j] * 127 / FT_SCALE;
        }
    }

    return output;
}

auto interleave_for_l1_sparsity(const decltype(raw_network::l1_weight)& input)
{
    auto output = std::make_unique<std::array<std::array<int16_t, FT_SIZE * L1_SIZE>, OUTPUT_BUCKETS>>();

#if defined(SIMD_ENABLED)
    // transpose and interleave the weights in blocks of 4 FT activations
    for (size_t i = 0; i < OUTPUT_BUCKETS; i++)
    {
        for (size_t j = 0; j < L1_SIZE; j++)
        {
            for (size_t k = 0; k < FT_SIZE; k++)
            {
                // we want something that looks like:
                //[bucket][nibble][output][4]
                (*output)[i][(k / 4) * (4 * L1_SIZE) + (j * 4) + (k % 4)] = input[i][j][k];
            }
        }
    }
#else
    for (size_t i = 0; i < OUTPUT_BUCKETS; i++)
    {
        for (size_t j = 0; j < L1_SIZE; j++)
        {
            for (size_t k = 0; k < FT_SIZE; k++)
            {
                (*output)[i][j * FT_SIZE + k] = input[i][j][k];
            }
        }
    }
#endif
    return output;
}

auto cast_l1_weight_int8(const std::array<std::array<int16_t, FT_SIZE * L1_SIZE>, OUTPUT_BUCKETS>& input)
{
    auto output = std::make_unique<std::array<std::array<int8_t, FT_SIZE * L1_SIZE>, OUTPUT_BUCKETS>>();

    for (size_t i = 0; i < OUTPUT_BUCKETS; i++)
    {
        for (size_t j = 0; j < FT_SIZE * L1_SIZE; j++)
        {
            (*output)[i][j] = static_cast<int8_t>(input[i][j]);
        }
    }

    return output;
}

auto cast_l1_bias_int32(const decltype(raw_network::l1_bias)& input)
{
    auto output = std::make_unique<std::array<std::array<int32_t, L1_SIZE>, OUTPUT_BUCKETS>>();

    for (size_t i = 0; i < OUTPUT_BUCKETS; i++)
    {
        for (size_t j = 0; j < L1_SIZE; j++)
        {
            (*output)[i][j] = static_cast<int32_t>(input[i][j]);
        }
    }

    return output;
}

auto transpose_l2_weights(const decltype(raw_network::l2_weight)& input)
{
    auto output = std::make_unique<std::array<std::array<std::array<float, L2_SIZE>, L1_SIZE * 2>, OUTPUT_BUCKETS>>();

    for (size_t i = 0; i < OUTPUT_BUCKETS; i++)
    {
        for (size_t j = 0; j < L1_SIZE * 2; j++)
        {
            for (size_t k = 0; k < L2_SIZE; k++)
            {
                (*output)[i][j][k] = input[i][k][j];
            }
        }
    }

    return output;
}

}

using namespace NN;

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cout << "Error: expected 2 command line arguments <input> and <output>" << std::endl;
        return EXIT_FAILURE;
    }

    if (auto size = std::filesystem::file_size(argv[1]); size != sizeof(raw_network))
    {
        std::cout << "Error: expected to read " << sizeof(raw_network) << " bytes but file size is " << size << " bytes"
                  << std::endl;
        return EXIT_FAILURE;
    }

    auto raw_net = std::make_unique<raw_network>();
    std::ifstream in(argv[1], std::ios::binary);
    in.read(reinterpret_cast<char*>(raw_net.get()), sizeof(raw_network));

    auto [ft_weight, ft_threat_weight, ft_bias, l1_weight]
        = shuffle_ft_neurons(raw_net->ft_weight, raw_net->ft_threat_weight, raw_net->ft_bias, raw_net->l1_weight);
    auto [ft_weight2, ft_threat_weight2, ft_bias2] = adjust_for_packus(*ft_weight, *ft_threat_weight, *ft_bias);
    auto final_net = std::make_unique<network>();
    final_net->ft_weight = *ft_weight2;
    final_net->ft_threat_weight = *ft_threat_weight2;
    final_net->ft_bias = *ft_bias2;
    final_net->l1_weight = *cast_l1_weight_int8(*interleave_for_l1_sparsity(*l1_weight));
    final_net->l1_bias = *cast_l1_bias_int32(*rescale_l1_bias(raw_net->l1_bias));
    final_net->l2_weight = *transpose_l2_weights(raw_net->l2_weight);
    final_net->l2_bias = raw_net->l2_bias;
    final_net->l3_weight = raw_net->l3_weight;
    final_net->l3_bias = raw_net->l3_bias;

    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(final_net.get()), sizeof(network));

    std::cout << "Created embedded network" << std::endl;
}
