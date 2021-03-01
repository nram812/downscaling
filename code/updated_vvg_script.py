import segmentation_models as sm
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


df = xr.open_dataset(r"C:\Users\user\OneDrive - NIWA\Research Projects\CDFP2102_Forecasting\nz_reanalysis_data_hourly.nc")
df_ = df.isel(latitude = slice(0,75), longitude = slice(0,75))

coarse_res = df_['t2m'][:,1,::8,::8]
coarse_img = df_['t2m'].interp_like(coarse_res)
# Created a coarse resolution input of the original field

# Lets now enhance the resolution of the field
reinterp = coarse_img.interp_like(df_['t2m'].isel(time =-10), method='nearest')
start =15
end = 32
X = reinterp.isel(expver =1, latitude = slice(start, start+end), longitude = slice(start, start+end))
y = df_['t2m'].isel(expver =1, latitude = slice(start, start+end), longitude = slice(start, start+end))
mask = np.where((np.sum(np.isnan(y), axis = (1,2)) ==0) & \
                (np.sum(np.isnan(X), axis = (1,2)) ==0))
X = X.isel(time = mask[0])
y = y.isel(time = mask[0])

X1= coarse_img.isel(time = mask[0]).interp_like(df_['t2m'].isel(time =-10), method = 'nearest').isel(expver =1, latitude = slice(start, start+end), longitude = slice(start, start+end))
def downscaling(x, y):
    f = linregress(x, y)
    return f.slope *x + f.intercept

downscaled = xr.apply_ufunc(downscaling, X1, y, input_core_dims=[["time"],["time"]],
               output_core_dims=[["time"]], dask='allowed',
               vectorize=True)
from sklearn.linear_model import LinearRegression

def downscaling(x, y):
    print(x.shape)
    x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
    cls =  LinearRegression()
    cls.fit(x, y)
    return cls.predict(x)

downscaled = xr.apply_ufunc(downscaling, X, y, input_core_dims=[["time"],["time"]],
               output_core_dims=[["time"]], dask='allowed',output_dtypes=[float],
               vectorize=True)



start = 2
plt.figure()
coarse_img.isel(time =-10, expver =1).isel(latitude = slice(start, start+64), longitude = slice(start, start+64)).plot(cmap ='RdBu_r',
                                                                                                                    vmin = 270, vmax = 295)
plt.show()


plt.figure()
reinterp.isel(time =-10, expver =1).isel(latitude =slice(start, start+64), longitude = slice(start, start+64)).plot(cmap ='RdBu_r',
                                                                                                                    vmin = 270, vmax = 295)
plt.savefig('coarse_res_image.png', dpi =300)
plt.show()

plt.figure()
downscaled.isel(time =-10).isel(latitude =slice(start, start+64), longitude = slice(start, start+64)).plot(cmap ='RdBu_r',
                                                                                                                    vmin = 270, vmax = 295)
plt.show()


plt.figure()
df_['t2m'].isel(time =-10, expver =1).isel(latitude =slice(start, start+64), longitude = slice(start, start+64)).plot(cmap ='RdBu_r',
                                                                                                                    vmin = 270, vmax = 295)
plt.savefig('high_res_image.png', dpi =300)
plt.show()



x_train_, x_test, y_train, y_test = train_test_split(np.repeat(X.values[:,:,:,np.newaxis],3,axis =-1), y.values, shuffle=False)

BACKBONE = 'vgg16'
# # load your data
# preprocess input
x_test = 255 * (x_test - x_train_.min())/ (x_train_.max() - x_train_.min())
y_test = 255 * (y_test - x_train_.min())/ (x_train_.max() - x_train_.min())
y_train = 255 * (y_train - x_train_.min())/ (x_train_.max() - x_train_.min())

x_train = 255 * (x_train_ - x_train_.min())/ (x_train_.max() - x_train_.min())

print(x_train.max(),x_test.max(), x_train.min(), x_test.min())




# define model
import tensorflow as tf
tens = tf.keras.layers.Input(shape=(32,32,3))
inputs = tf.keras.applications.vgg16.preprocess_input(
    tens, data_format=None
)
outs = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape = (32,32,3),
               encoder_freeze = True, activation ='linear',
               decoder_use_batchnorm =True, decoder_block_type = 'transpose')(inputs)
model = tf.keras.models.Model(tens, outs)
model.compile(
    'adam',
    loss='mae',
)
dot_img_file = 'model_1.png'
tf.keras.utils.plot_model(model2, to_file=dot_img_file, show_shapes=True)

# fit model
# if you use data generator use model.fit_generator(...) instead of model.fit(...)
# more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
model.fit(
   x=x_train,
   y=y_train,
   batch_size=40,validation_data = (x_test, y_test),
   epochs=330,
shuffle = True)
preds= model.predict(x_test)
plt.figure()
preds1 = preds * (x_train_.max() - x_train_.min())/255.0 +  x_train_.min()
ytest1 = y_test * (x_train_.max() - x_train_.min())/255.0 +  x_train_.min()
xtest1 = x_test * (x_train_.max() - x_train_.min())/255.0 +  x_train_.min()
fig, ax = plt.subplots(1,3, figsize = (15,8))
index = 375
ax[0].imshow(xtest1[index,:,:,0], vmin =270, vmax =290, cmap ='RdBu_r')
ax[0].set_title('Input Image')
ax[1].imshow(preds1[index][:,:,0], vmin =270, vmax =290, cmap ='RdBu_r')
ax[1].set_title('DL Downscaled')
cs = ax[2].imshow(ytest1[index,:,:], vmin =270, vmax =290, cmap ='RdBu_r')
ax[2].set_title('Original Image')
ax4 = fig.add_axes([0.1, 0.1, 0.8, 0.03])
cbar = fig.colorbar(cs, cax = ax4, orientation = 'horizontal')
cbar.set_label('2 Meter Temperature (K)')
fig.savefig('Downscale_exampled4.png', dpi=300)
fig.show()


plt.figure()
plt.imshow(np.sqrt(np.nanmean(abs(ytest1 - preds1[:,:,:,0])**2, axis =0)), vmin =0, vmax =2)
plt.colorbar()
plt.show()



from scipy.stats import linregress
f = linregress(preds[:,5,29,0],y_test[:,5,29])

print(f)
f = linregress(x_test[:,5,29,0],y_test[:,5,29])
print(f)



tens = tf.keras.layers.Input(shape=(128,128,3))
inputs = tf.keras.applications.resnet.preprocess_input(
    tens, data_format=None
)
outs2 = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape = (128,128,3),
               encoder_freeze = True, activation ='linear',
               decoder_use_batchnorm =False, decoder_block_type = 'transpose')(inputs)
model2 = tf.keras.models.Model(tens, outs2)
model2.compile(
    'rmsprop',
    loss='mse',
)

model2.load_weights('deep_encoderweights.h5')

model2 = tf.keras.models.load_model('vgg16.h5')

d = {}
d['latitude'] = (('latitude'), np.linspace(reinterp.latitude.min(), reinterp.latitude.max(), 128))
d['longitude'] = (('longitude'), np.linspace(reinterp.longitude.min(), reinterp.longitude.max(), 128))
d['data'] = (('latitude','longitude'), np.zeros([128,128]))
dset = xr.Dataset(d)

test_ = df_['t2m'].interp_like(dset).isel(time = mask[0], expver=1)
test_ = 255 * (test_ - x_train_.min())/ (x_train_.max() - x_train_.min())
preds_ = model2.predict(np.repeat(test_.values[:,:,:,np.newaxis],3,axis =-1)[0:50])

plt.figure()
plt.imshow(preds_[-20][:,:,0], vmin =10, vmax =185, cmap ='RdBu')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(test_[-20,:,:], vmin =10, vmax =185, cmap ='RdBu')
plt.colorbar()
plt.show()

