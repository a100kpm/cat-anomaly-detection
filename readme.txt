The repository showcase 3 different deeplearning architectures (autoencoder, variational autoencoder and wasserstein generative adversial network)
used to detect anomalous data in dataset. In the present case, find all non cat image in a cat dataset from kaggle.
There is no label.

To run the files properly, follow instructions in data_treat_cat.py
autoencoder file and variational autoencoder file showcase a full solution to the problem. 
Result may vary due to the random split of the dataset in train validation and test that i did without a seed.

The files showcase a lot of comments remnant of thought process to reach the result (especially in the model_cat_find_anomalies files)
Although the projet is turned arround finding anomalous data, the 3 architectures are capable of becoming image generators, assuming you train them long enough toward that goal.

Once you trained a model, you should be able to run without adjustment the associated file model_cat_find_anomalies and read the anomalous data directly in the variable "df".
Variables dff in autoencoder_cat_find_anomalies.py and df_test in vae_cat_find_anomalies.py are "fine tuned" dataframe for my specific case.
They are showcasing the possibility to lineary separate a good chunk of anomalous data along the created features. (reconstruction error, kde, gaussian mixture etc...).


As a brief summary of my result:
Autoencoder did decent for anomaly detection (6 anomalies found, see autoencoder_anomalies.png), 
and decent for image reconstruction both on legitimate and anomalous images (see autoencoder_reconstruct_normal.png and autoencoder_reconstruct_anomalies.png).
Training was >400 epochs

Variational autoencoder did exeptionaly well for anomaly detection (8 anomalies found, see vae_anomalies.png),
and terrible for image reconstruction (see vae_reconstruct_normal.png and vae_reconstruct_anomalies.png).
Training was 40 epochs (with bat choice of beta_vae and questionable jump of learning rate if you look at the file).
If you want to train it to reconstruct images properly, remember to start the training with low value (or 0) for the parameter beta_vae, and to increase it slowly over time.

Wasserstein generative adversarial network did not yield good result, the training is slow (expect at least >2000 epochs to start having something that look like a reconstruction of image)
And pay attention to not fall in collapse mode (parameters in wgan.py should not create a collapse mode for at least 800 epochs).
If in the future I get results with the wgan, I'll add a wgan_find_anomalies.py file.

Note: the division in train, validation and test of the dataset is not needed for the task, we can however use them to observe overfit on the training set, using different tool (like kde)
which will yield similar result on validation and test, but not on train. 
It is especially remarkable because by using val_loss and train_loss the usual way to see overfitting, 
we can arrive at the conclusion that the ae is soon to be overfitting while it has actually already overfitted.

