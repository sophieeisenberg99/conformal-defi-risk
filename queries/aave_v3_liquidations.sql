-- Aave v3 liquidation events on Ethereum mainnet (Dune SQL).
-- Produced data/aave_v3_liquidations.csv (104 events). Run on dune.com, export as CSV.
--
-- Notes
-- * Table: aave_v3_ethereum.pool_evt_liquidationcall (decoded LiquidationCall event).
--   The earliest event in this table is 2023-01-30, so earlier windows return no rows.
-- * Amounts are raw token units divided by 1e18. That is only correct for 18-decimal
--   tokens (WETH, DAI, AAVE); USDC/USDT use 6 and WBTC 8, and nothing is converted to
--   USD. Summing these across assets mixes units. Joining prices.* and token decimals
--   (or, for the target, converting to USD) is the main data limitation of this export.
-- * The event has no health factor, so that column is not exported.

select
    evt_block_time                     as block_time,
    liquidator                         as liquidator,
    user                               as user,
    collateralAsset                    as collateral_asset,
    debtAsset                          as debt_asset,
    liquidatedCollateralAmount / 1e18  as collateral_amount,
    debtToCover / 1e18                 as debt_repaid
from aave_v3_ethereum.pool_evt_liquidationcall
where evt_block_time >= timestamp '2023-01-30'
  and evt_block_time <  timestamp '2023-06-30'
order by evt_block_time asc
