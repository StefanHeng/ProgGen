import src.generate.step_wise.util_entity_correct as uec

__all__ = ['CORRECT', 'WRONG_SPAN', 'WRONG_TYPE', 'NA']

CORRECT = uec.LABEL_CORRECT
WRONG_SPAN = uec.LABEL_WRONG_BOUNDARY
WRONG_TYPE = uec.LABEL_WRONG_TYPE
NA = uec.LABEL_NOT_NAMED_ENTITY
