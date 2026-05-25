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
        251, 170, 116, 303, 157, 230, 193, 160, 202, 41, 124, 154, 62, 143, 234, 315, 123, 76, 49, 44, 208, 294, 21, 92, 94, 221, 266, 70, 166, 245, 65, 153, 277, 279, 289, 235, 167, 53, 98, 259, 268, 79, 309, 292, 134, 310, 270, 201, 306, 316, 23, 4, 318, 13, 216, 100, 112, 280, 175, 18, 236, 156, 212, 12, 52, 93, 28, 147, 263, 78, 113, 43, 135, 177, 6, 74, 108, 33, 102, 267, 241, 313, 233, 291, 75, 207, 84, 265, 81, 271, 71, 73, 319, 82, 31, 281, 276, 91, 225, 99, 227, 237, 244, 57, 314, 226, 138, 46, 296, 104, 106, 152, 248, 110, 307, 144, 242, 255, 85, 39, 194, 262, 311, 35, 48, 256, 125, 139, 80, 122, 60, 218, 158, 63, 283, 114, 284, 101, 109, 86, 61, 168, 51, 128, 136, 141, 223, 239, 148, 209, 164, 15, 115, 97, 55, 151, 224, 215, 243, 5, 132, 36, 117, 195, 285, 129, 121, 179, 162, 89, 131, 1, 299, 165, 22, 127, 305, 50, 32, 47, 68, 293, 182, 205, 184, 142, 118, 304, 246, 20, 120, 274, 317, 29, 2, 186, 19, 37, 196, 146, 200, 240, 54, 133, 203, 171, 197, 173, 38, 282, 252, 211, 189, 150, 300, 206, 302, 187, 8, 278, 159, 69, 222, 214, 130, 169, 219, 137, 64, 42, 172, 210, 183, 67, 298, 34, 111, 40, 145, 258, 192, 140, 181, 17, 238, 176, 188, 59, 25, 249, 250, 56, 163, 301, 126, 96, 257, 95, 220, 191, 107, 103, 90, 185, 180, 178, 204, 253, 217, 269, 14, 161, 190, 254, 229, 273, 174, 232, 66, 155, 228, 312, 27, 83, 88, 9, 295, 287, 30, 275, 290, 3, 26, 199, 45, 10, 272, 297, 198, 119, 247, 308, 288, 261, 58, 286, 0, 260, 231, 213, 11, 16, 77, 24, 105, 7, 264, 149, 87, 72,
    };
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
