import segmentation_models_pytorch as smp

def get_model(encoder_name='resnet34', encoder_weights='imagenet'):
    """
    Retourne un modèle U-Net avec backbone pré-entraîné sur ImageNet.
    Sortie sigmoïde pour segmentation binaire.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )
    return model