'''
Author: Peng Bo
Date: 2022-11-22 10:44:02
LastEditTime: 2022-11-22 10:45:23
Description: 

'''

from  collections import OrderedDict

DLIB_68_PTS_MODEL_IDX = {
	"jaw" : list(range(0, 17)),
	"left_eyebrow" : list(range(17,22)),
	"right_eyebrow" : list(range(22,27)),
	"nose" : list(range(27,36)),
	"left_eye" : list(range(36, 42)),
	"right_eye" : list(range(42, 48)),
	"left_eye_poly": list(range(36, 42)),
	"right_eye_poly": list(range(42, 48)),
	"mouth" : list(range(48,68)),
	"eyes" : list(range(36, 42))+list(range(42, 48)),
	"eyebrows" : list(range(17,22))+list(range(22,27)),
	"eyes_and_eyebrows" : list(range(17,22))+list(range(22,27))+list(range(36, 42))+list(range(42, 48)),
}

WFLW_98_PTS_MODEL_IDX = {
	"jaw" : list(range(0,33)),
	"left_eyebrow" : list(range(33,42)),
	"right_eyebrow" : list(range(42,51)),
	"nose" : list(range(51, 60)),
	"left_eye" : list(range(60, 68))+[96],
	"right_eye" : list(range(68, 76))+[97],
	"left_eye_poly": list(range(60, 68)),
	"right_eye_poly": list(range(68, 76)),
	"mouth" : list(range(76, 96)),
	"eyes" : list(range(60, 68))+[96]+list(range(68, 76))+[97],
	"eyebrows" : list(range(33,42))+list(range(42,51)),
	"eyes_and_eyebrows" : list(range(33,42))+list(range(42,51))+list(range(60, 68))+[96]+list(range(68, 76))+[97],
}

DLIB_68_TO_WFLW_98_IDX_MAPPING = OrderedDict()
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(0,17),range(0,34,2)))) # jaw | 17 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(17,22),range(33,38)))) # left upper eyebrow points | 5 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(22,27),range(42,47)))) # right upper eyebrow points | 5 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(27,36),range(51,60)))) # nose points | 9 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({36:60}) # left eye points | 6 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({37:61})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({38:63})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({39:64})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({40:65})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({41:67})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({42:68}) # right eye | 6 pts
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({43:69})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({44:71})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({45:72})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({46:73})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update({47:75})
DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(48,68),range(76,96)))) # mouth points | 20 pts
WFLW_98_TO_DLIB_68_IDX_MAPPING = {v:k for k,v in DLIB_68_TO_WFLW_98_IDX_MAPPING.items()}