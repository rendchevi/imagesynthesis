def weight_clipping(model, clip_value = (-0.01, 0.01)):

    for layer in model.layers:
        # Get layer weights
        weights = layer.get_weights()
        # Clip value 
        clipped_weights = []
        if len(weights) > 0:
            for weight in weights:
                weight[weight < clip_value[0]] = clip_value[0]
                weight[weight > clip_value[1]] = clip_value[1]
                clipped_weights.append(weight)
            # Update weights values
            layer.set_weights(clipped_weights)

    return None