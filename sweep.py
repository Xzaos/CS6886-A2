import subprocess
import sys

if __name__ == '__main__':
    weight_bits_list = [8, 6, 4, 3, 2]
    act_bits_list = [8, 6, 4]
    total = len(weight_bits_list) * len(act_bits_list)
    i = 0
    for wb in weight_bits_list:
        for ab in act_bits_list:
            print(f'Running w{wb}a{ab} ({i+1}/{total})')
            result = subprocess.run([
                sys.executable, 'test.py',
                '--weight_bits', str(wb),
                '--act_bits', str(ab),
                '--use_gptq',
                '--checkpoint', 'checkpoints/mobilenetv2_cifar10.pth',
            ])
            if result.returncode != 0:
                print(f'Warning: w{wb}a{ab} failed with returncode {result.returncode}')
            i += 1
