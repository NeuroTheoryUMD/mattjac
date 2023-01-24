import itertools as it

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

