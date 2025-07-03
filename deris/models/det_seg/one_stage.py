from deris.models import MODELS, build_vis_enc, build_head
from .base import BaseModel


@MODELS.register_module()
class OneStageModel(BaseModel):
    def __init__(self, vis_enc, head):
        super(OneStageModel, self).__init__()
        self.vis_enc = build_vis_enc(vis_enc)
        if head is not None:
            self.head = build_head(head)
