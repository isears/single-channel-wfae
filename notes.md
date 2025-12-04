# Notes

## TODOs
- Give channel to encoder / decoder (can be encoded as 3 bits)
- Ensure consistent validation sets (training and classification assessment)

## 11/22/25
- Changed latent dim from 64 -> 16, seems like MSE loss gets down to 0.010 (at 64 had been 0.011)
- Increase filters to 16, decrease latent dim even further to 10: MSE down to 0.0070 (15 epochs)
- Filters 16, decrease latent dim to 3: MSE 0.009 in 7 epochs
- Filters 16, decrease latent dim to 2: MSE 0.007 in 16 epochs
- Filters 32, latent dim 2: MSE 0.005 in 18 epochs
- Seems like after MSE 0.007, label prediction doesn't necessarily get better