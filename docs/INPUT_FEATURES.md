# Final model inputs

This table documents the final model inputs. Factor names, ordering, units,
classifications, and field labels reproduce the structured prompt implemented
in `src/utils.py`. A dash means that the prompt does not state a unit for that
input.

| No. | Input | Unit | Temporal resolution | Classification | Exact prompt representation |
|---:|---|---|---|---|---|
| - | Historical UHI Intensity | °C | Monthly; previous 12 months | Temporal context | `[UHI Data:]` |
| 1 | Area | km² | Static | Structural | `[Area:]` |
| 2 | Built Height | m | Static | Structural | `[Built Height:]` |
| 3 | Built Surface Area | m²/km² | Every five years | Structural | `[Built Surface Area:]` |
| 4 | Urbanization Level | - | Every five years | Structural | `[Urbanization Level:]` |
| 5 | Location | - | Static | Structural | `[Location:]` |
| 6 | City Compactness | - | Static | Structural | `[City Compactness:]` |
| 7 | Population Density | people/km² | Every five years | Structural | `[Population Density:]` |
| 8 | Urban Land Use | % | Static | Structural | `[Urban Land Use:]` |
| 9 | Average Cloud Cover | - | Monthly; previous 12 months | Liquidity | `[Average Cloud Cover:]` |
| 10 | Solar Radiation | J m⁻² d⁻¹ | Monthly; previous 12 months | Liquidity | `[Solar Radiation:]` |
| 11 | Precipitation Flux | mm/day | Monthly; previous 12 months | Liquidity | `[Precipitation Flux:]` |
| 12 | Snow Thickness | m | Monthly; previous 12 months | Liquidity | `[Snow Thickness:]` |
| 13 | Enhanced Vegetation Index (EVI) | - | Monthly; previous 12 months | Liquidity | `[Enhanced Vegetation Index (EVI):]` |
| 14 | Normalized Difference Vegetation Index (NDVI) | - | Monthly; previous 12 months | Liquidity | `[Normalized Difference Vegetation Index (NDVI):]` |
| 15 | PM2.5 Concentration | μg/m³ | Monthly; previous 12 months | Liquidity | `[PM2.5 Concentration:]` |
| 16 | Average Temperature | K | Monthly; previous 12 months | Liquidity | `[Average Temperature:]` |
| 17 | Average Vapour Pressure | hPa | Monthly; previous 12 months | Liquidity | `[Average Vapour Pressure:]` |
| 18 | Average Wind Speed | m/s | Monthly; previous 12 months | Liquidity | `[Average Wind Speed:]` |

## UHI Knowledge Enhancer ablation input

The `w/o UHI Knowledge Enhancer` variant removes all section headings, field
labels, prose, and units. It serializes the same 147 input values as one list:

```text
[target_year, target_month, historical_uhi_1, ..., wind_speed_12]
```
