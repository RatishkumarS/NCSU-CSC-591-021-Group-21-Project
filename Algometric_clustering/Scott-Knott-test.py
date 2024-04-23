from raise_utils.interpret import ScottKnott

data = {
    "K-Means": [
        0.1343778530974283,
        0.1343778530974283,
        0.9054327502869586,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
        0.8634424423754842,
    ],
    "SVM": [
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
    ],
    "SMO-GGM": [
        0.150361705504544,
        0.16637918671853072,
        0.18916209503810977,
        0.1677989951476396,
        0.17444206028279968,
        0.18460262607015523,
        0.20306827601440017,
        0.13412380775106938,
        0.1366030469287337,
        0.18170903026607715,
        0.17353705174750347,
        0.14223167323108896,
        0.16598230444086365,
        0.17205253165980486,
        0.1565670473391886,
        0.16115391857798633,
        0.13802498577876499,
        0.18230089165713279,
        0.19658132320673444,
        0.17008796011115643,
    ],
    "Algometric-Clustering": [
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
        0.1343778530974283,
    ],
    "SMO": [
        0.6288925747787598,
        0.6056630321213916,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
        0.38583077364562396,
    ],
}


sk = ScottKnott(data)
sk.pprint()
