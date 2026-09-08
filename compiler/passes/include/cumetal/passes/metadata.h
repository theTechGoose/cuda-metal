#pragma once

#include "cumetal/common/air_toolchain.h"

#include "cumetal/ptx/parser.h"

#include <string>
#include <vector>

namespace cumetal::passes {

struct MetadataField {
    std::string key;
    std::string value;
};

struct KernelMetadata {
    std::string kernel_name;
    std::vector<MetadataField> fields;
};

struct MetadataOptions {
    // Default to the dialect the installed Metal toolchain accepts; an explicit
    // value still wins. See cumetal::common::detected_air_dialect().
    std::string air_version = cumetal::common::detected_air_dialect().air_version_string();
    std::string language_version =
        cumetal::common::detected_air_dialect().language_version_string();
};

KernelMetadata build_kernel_metadata(const cumetal::ptx::EntryFunction& entry,
                                     const MetadataOptions& options = {});

}  // namespace cumetal::passes
