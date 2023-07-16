from .models import make as make_model

def get_model_for_finetuning(name, num_classes):
    return make_model(name, num_classes)