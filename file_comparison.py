import numpy as np

_path = './classification/'

n_neurons = int(400)
time = int(64)

pathBindsOut = 'labelsBindsnetOut'+str(n_neurons)+'N_'+str(time)+'ms.csv'

# Labels Comparison
fb_1 = np.fromfile( _path + 'labelsBindsnetIn'+str(n_neurons)+'N_'+str(time)+'ms.csv', dtype = np.int8 ,sep='\n')
fb_2 = np.fromfile( _path + 'labels_CppSerial_QT_'+str(n_neurons)+'N_64ms.csv', dtype = np.int8 ,sep='\n')
fb_3 = np.fromfile( _path + 'labelsBindsnetOut'+str(n_neurons)+'N_'+str(time)+'ms.csv', dtype = np.int8 ,sep='\n')

cmp1 = 0

tam = len(fb_2) if len(fb_2) < len(fb_1) else len(fb_1)
print('tam={}'.format(tam))
for indx in range(tam):
    if( fb_1[indx] == fb_2[indx] ):
        cmp1 += 1
    else:
        if indx < 100:
            print('indx [{}] - real [{}] - inference [{}] - Bindsnet Out [{}]'.format(indx+1, fb_1[indx], fb_2[indx], fb_3[indx]))
        
cmp1 = cmp1 / tam        
cmp2 = np.count_nonzero(fb_1 == fb_3) / len(fb_3)

print('{0:=>50}'.format(''))
print( 'Bindsnet In <--> Qt -> {0:2.2f} % for {1} samples'.format(cmp1*100, tam) )
print('{0:=>50}'.format(''))
print( 'Bindsnet Labels In <--> Bindsnet Labels Out -> {0:2.2f} %'.format(cmp2*100) )
print('{0:=>50}'.format(''))