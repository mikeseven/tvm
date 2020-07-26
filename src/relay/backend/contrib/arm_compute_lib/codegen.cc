/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/backend/contrib/arm_compute_lib/codegen.cc
 * \brief Implementation of the Relay -> ACL JSON serializer.
 */
#include <tvm/ir/module.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/type.h>

#include <memory>
#include <string>
#include <vector>

#include "../../utils.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief Generates an ACLModule from a relay expression. This "compilation"
 * does not require ACL since the actual conversion using ACL APIs is
 * deferred until creation of the runtime. This step simply serializes the
 * relay program into a JSON string.
 */
class ACLJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  ACLJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {}

  /*!
   * \brief Visit call nodes and generate appropriate JSON node.
   *
   * \param cn The current call node.
   * \return A list of graph entry nodes.
   */
  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    if (cn->op.as<OpNode>()) {
      return JSONSerializer::VisitExpr_(cn);
    }
    if (!cn->op.as<FunctionNode>()) {
      LOG(FATAL) << "Arm Compute Library JSON runtime does not support calls to "
                 << cn->op->GetTypeKey();
    }
    auto fn = cn->op.as<FunctionNode>();
    auto comp = fn->GetAttr<String>(attr::kComposite);
    CHECK(comp.defined()) << "Arm Compute Library JSON runtime only supports composite functions.";
    const std::string name = comp.value();
    std::shared_ptr<JSONGraphNode> json_node;
    if (name == "arm_compute_lib.conv2d") {
      json_node = CreateCompositeConvJSONNode(cn);
    } else {
      LOG(FATAL) << "Unrecognized Arm Compute Library pattern: " << name;
    }
    return AddNode(json_node, GetRef<Expr>(cn));
  }

 private:
  /*!
   * \brief Create a JSON representation of a composite convolution.
   *
   * \param call The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeConvJSONNode(const CallNode* cn) {
    const std::string name = "nn.conv2d";
    const CallNode* pad = nullptr;
    const CallNode* conv = nullptr;
    const CallNode* bias = nullptr;
    bool has_activation = false;

    // Unpack composite function
    const auto* fn = cn->op.as<FunctionNode>();
    CHECK(fn);
    const auto* current_call = fn->body.as<CallNode>();
    if (backend::IsOp(current_call, "nn.relu")) {
      has_activation = true;
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "nn.bias_add")) {
      bias = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    CHECK(backend::IsOp(current_call, "nn.conv2d"));
    conv = current_call;
    if (!current_call->args.empty() && current_call->args[0]->IsInstance<CallNode>()) {
      current_call = current_call->args[0].as<CallNode>();
      if (backend::IsOp(current_call, "nn.pad")) {
        pad = current_call;
      }
    }

    const auto* conv_attr = conv->attrs.as<Conv2DAttrs>();
    CHECK(conv_attr);
    CHECK(conv_attr->kernel_layout == "OHWI")
        << "Kernel layout must be OHWI, has the module been pre-processed correctly?";

    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(conv->args[1])[0]);
    if (bias) {
      inputs.push_back(VisitExpr(bias->args[1])[0]);
    }

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, conv);

    // Override attributes
    if (pad) {
      const auto* pad_attr = pad->attrs.as<PadAttrs>();
      CHECK(pad_attr);
      auto p = pad_attr->pad_width;
      // Convert to TVM layout for now, conversion to ACL layout takes place in runtime.
      // Standard convolution pad layout for TVM: top, left, bottom, right.
      std::vector<std::string> padding = {std::to_string(p[1][0].as<IntImmNode>()->value),
                                          std::to_string(p[2][0].as<IntImmNode>()->value),
                                          std::to_string(p[1][1].as<IntImmNode>()->value),
                                          std::to_string(p[2][1].as<IntImmNode>()->value)};
      std::vector<dmlc::any> padding_attr;
      padding_attr.emplace_back(padding);
      json_node->SetAttr("padding", padding_attr);
    }
    if (has_activation) {
      std::vector<std::string> activation_type = {"relu"};
      std::vector<dmlc::any> act_attr;
      act_attr.emplace_back(activation_type);
      json_node->SetAttr("activation_type", act_attr);
    }
    return json_node;
  }
};

/*!
 * \brief Pre-process a module containing functions ready for ACL codegen.
 *
 * For now we enforce OHWI kernel layout and fold the transforms away.
 *
 * \param mod The module to be pre-processed.
 * \return The processed module.
 */
IRModule PreProcessModule(const IRModule& mod) {
  IRModule preprocessed_module;
  tvm::Map<String, Array<String>> desired_layouts = {{"nn.conv2d", {"NHWC", "OHWI"}}};
  preprocessed_module = transform::ConvertLayout(desired_layouts)(mod);
  preprocessed_module = transform::FoldConstant()(preprocessed_module);
  return preprocessed_module;
}

TVM_REGISTER_GLOBAL("relay.ext.arm_compute_lib.optimize").set_body_typed(PreProcessModule);

/*!
 * \brief Create a runtime module for ACL.
 *
 * This consists of a series of "serialized functions" which each represent a
 * sub-graph to be computed by ACL and will each be executed independently from
 * one another. Each function consists of serialized JSON describing the sub-graph
 * and serialized constant tensors.
 *
 * \note The ACL runtime module only supports a single operator per
 * sub-graph currently.
 *
 * \param ref The ext_func Relay expression/module to be executed using extern ops.
 * \return A runtime module.
 */
runtime::Module ACLCompiler(const ObjectRef& ref) {
  CHECK(ref->IsInstance<FunctionNode>()) << "The input ref is expected to be a Relay function.";
  Function func = Downcast<Function>(ref);
  std::string func_name = backend::GetExtSymbol(func);

  ACLJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto param_names = serializer.GetParams();
  const auto* pf = runtime::Registry::Get("runtime.arm_compute_lib_runtime_create");
  CHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  runtime::Module lib = (*pf)(func_name, graph_json, param_names);
  return lib;
}

TVM_REGISTER_GLOBAL("relay.ext.arm_compute_lib").set_body_typed(ACLCompiler);

/*!
 * \brief Check whether ACL graph runtime is used.
 *
 * \return True if ACL graph runtime is enabled, False if not.
 */
inline constexpr bool IsACLRuntimeEnabled() {
#if TVM_GRAPH_RUNTIME_ARM_COMPUTE_LIB
  return true;
#else
  return false;
#endif
}

TVM_REGISTER_GLOBAL("relay.op.is_arm_compute_runtime_enabled").set_body_typed(IsACLRuntimeEnabled);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
