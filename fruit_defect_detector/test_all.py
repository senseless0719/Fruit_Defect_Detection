#!/usr/bin/env python3
"""Quick test script to validate all predictions."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from predict import load_models, predict_image

p0, p1, p2 = load_models('./models')

tests = [
    ('E:/DOWNLOADSinit/icelands-ring-road-endless-road-landscape-snow-covered-3840x2160-3922.jpg', 'NOT_FRUIT'),
    ('E:/DOWNLOADSinit/astronaut-space-cosmos-planet-rings-of-saturn-orbit-surreal-3840x2160-6739.jpg', 'NOT_FRUIT'),
    ('E:/DOWNLOADSinit/yasuke-dark-7680x4320-21997.jpg', 'NOT_FRUIT'),
    ('E:/DOWNLOADSinit/try1.jpg', 'DEFECTIVE'),
    ('E:/DOWNLOADSinit/try2.jpeg', 'DEFECTIVE'),
    ('E:/DOWNLOADSinit/try3.jpg', 'DEFECTIVE'),
    ('./dataset/healthy/fn_good_00050.jpg', 'HEALTHY'),
    ('./dataset/scab/pv_scab_00000.jpg', 'DEFECTIVE'),
    ('./dataset/black_rot/pv_blackrot_00000.jpg', 'DEFECTIVE'),
    ('./dataset/cedar_rust/pv_cedarrust_00000.jpg', 'DEFECTIVE'),
    ('./dataset/full_damage/fn_bad_00000.jpg', 'DEFECTIVE'),
]

passed = 0
total = 0
for path, expected in tests:
    name = os.path.basename(path)[:45]
    if not os.path.exists(path):
        print(f"  SKIP  {name} (file not found)")
        continue
    total += 1
    r = predict_image(path, p0, p1, p2)
    is_fruit = r.get('is_fruit', False)
    defect = r.get('defect_type', 'N/A')
    conf = r.get('confidence', 0)

    if not is_fruit:
        got = 'NOT_FRUIT'
    elif r.get('defect_found'):
        got = f'DEFECTIVE ({defect} {conf:.0f}%)'
    else:
        got = f'HEALTHY ({conf:.0f}%)'

    ok = (expected == 'NOT_FRUIT' and got == 'NOT_FRUIT') or \
         (expected == 'DEFECTIVE' and 'DEFECTIVE' in got) or \
         (expected == 'HEALTHY' and 'HEALTHY' in got)
    if ok:
        passed += 1
    mark = 'PASS' if ok else 'FAIL'
    print(f"  [{mark}]  {name:45s}  exp={expected:10s}  got={got}")

print(f"\n  Result: {passed}/{total} passed")
