import copy
import itertools as it


# converts each layer param into a list to make combinatorics easier
def _listify(model_template):
    listified_model_template = copy.deepcopy(model_template)
    for network in listified_model_template.networks:
        for layer in network.layers:
            for param in layer.params:
                if isinstance(layer.params[param], )
            
            layer.params = {
                'internal_layer_type': [layer.params['internal_layer_type']],
                'num_filters': num_filters if isinstance(num_filters, list) else [num_filters],
                'num_inh': num_inh if isinstance(num_inh, list) else [num_inh],
                'bias': bias if isinstance(bias, list) else [bias],
                'norm_type': [nt.value for nt in norm_type] if isinstance(norm_type, list) else [norm_type.value],
                'NLtype': [nl.value for nl in NLtype] if isinstance(NLtype, list) else [NLtype.value],
                'initialize_center': initialize_center if isinstance(initialize_center, list) else [initialize_center],
                'reg_vals': reg_vals if isinstance(reg_vals, list) else [reg_vals],
                'output_norm': output_norm if isinstance(output_norm, list) else [output_norm],
                'filter_dims': [filter_dims] if not isinstance(filter_dims, list) else filter_dims,
                'window': window if isinstance(window, list) else [window],
                'padding': padding if isinstance(padding, list) else [padding],
                'pos_constraint': pos_constraint if isinstance(pos_constraint, list) else [pos_constraint]
            }
    return listified_model_template


def sequential(model_template):
    ...


####### Params Exploder
def grid(model_template):
    # TODO: convert model_template layer params to lists here
    
    networks_with_exploded_layers = []
    for network in model_template.networks:
        # break apart the layer definitions into
        # a list of each possible combination of parameters
        # so each layer definition like:
        #  [ [subs=[1,2], filter=[3,4],  [subs=[a,b], filter=[c,d] ]
        #  -->  [
        #       [[subs=1, filter=3], [subs=1, filter=4],
        #        [subs=2, filter=3], [subs=2, filter=4]],
        #
        #       [[subs=a, filter=b], [subs=a, filter=d],
        #        [subs=b, filter=e], [subs=b, filter=d]] 
        exploded_layers = [list(it.product(*layer.build())) for layer in network.layers]

        # break apart the list of each possible combination of parameters
        # 
        # so each layer definition like:
        #  [ [subs=[1,2], filter=[3,4],  [subs=[a,b], filter=[c,d] ]
        #  -->  [
        #       [[subs=1, filter=3], [subs=1, filter=4],
        #        [subs=2, filter=3], [subs=2, filter=4]],
        #
        #       [[subs=a, filter=b], [subs=a, filter=d],
        #        [subs=b, filter=c], [subs=b, filter=d]]
        #    ]
        #        
        #    --> [
        #         [[subs=1, filter=3], [subs=a, filter=b]],
        #         [[subs=1, filter=3], [subs=a, filter=d]],
        #         [[subs=1, filter=3], [subs=b, filter=c]],
        #         [[subs=1, filter=3], [subs=b, filter=d]],
        #         ...
        #         [[subs=2, filter=4], [subs=a, filter=b]],
        #         [[subs=2, filter=4], [subs=a, filter=d]],
        #         [[subs=2, filter=4], [subs=b, filter=c]],
        #         [[subs=2, filter=4], [subs=b, filter=d]],
        #        ]
        #           
        layer_groups = list(it.product(*exploded_layers))
        networks_with_exploded_layers.append(layer_groups)

    # each network has a (list of layer.configuration_lists)
    # so we need to cross-product the possible layers in the networks
    # each 
    #                 [network.(list of layer.configuration_lists)]
    #     Network[ 
    #               Layer[
    #                     [subs=1, filter=3], [subs=a, filter=c]],
    #                     [subs=1, filter=3], [subs=a, filter=d]],
    #                    ]
    #             ],
    #     Network[
    #               Layer[
    #                     [subs=1, filter=3], [subs=a, filter=c]],
    #                     [subs=1, filter=3], [subs=a, filter=d]],
    #                    ]
    #             ]
    #     ...
    #         ]
    model_configurations = list(it.product(*networks_with_exploded_layers))
    return model_configurations

