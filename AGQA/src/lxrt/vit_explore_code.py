import timm

m = timm.create_model('mobilenetv3_large_100', pretrained=True)
m.eval()
from pprint import pprint
model_names = timm.list_models(pretrained=True)
pprint(model_names)