CREATE TABLE financial.nz_dollar(
	dealer_long INTEGER NOT NULL,
	dealer_short INTEGER NOT NULL,
	dealer_spreading INTEGER NOT NULL,

	institutional_long INTEGER NOT NULL,
	institutional_short INTEGER NOT NULL,
	institutional_spreading INTEGER NOT NULL,

	leveragedfunds_long INTEGER NOT NULL,
	leveragedfunds_short INTEGER NOT NULL,
	leveragedfunds_spreading INTEGER NOT NULL,

	other_long INTEGER NOT NULL,
	other_short INTEGER NOT NULL,
	other_spreading INTEGER NOT NULL,

	dealer_long_change INTEGER NOT NULL,
	dealer_short_change INTEGER NOT NULL,
	dealer_spreading_change INTEGER NOT NULL,

	institutional_long_change INTEGER NOT NULL,
	institutional_short_change INTEGER NOT NULL,
	institutional_spreading_change INTEGER NOT NULL,

	leveragedfunds_long_change INTEGER NOT NULL,
	leveragedfunds_short_change INTEGER NOT NULL,
	leveragedfunds_spreading_change INTEGER NOT NULL,

	other_long_change INTEGER NOT NULL,
	other_short_change INTEGER NOT NULL,
	other_spreading_change INTEGER NOT NULL,

	report_date DATE NOT NULL,
	id SERIAL PRIMARY KEY
);

CREATE TABLE metals.silver(
	producer_long INTEGER NOT NULL,
	producer_short INTEGER NOT NULL,

	swapdealers_long INTEGER NOT NULL,
	swapdealers_short INTEGER NOT NULL,
	swapdealers_spreading INTEGER NOT NULL,

	managedmoney_long INTEGER NOT NULL,
	managedmoney_short INTEGER NOT NULL,
	managedmoney_spreading INTEGER NOT NULL,

	other_long INTEGER NOT NULL,
	other_short INTEGER NOT NULL,
	other_spreading INTEGER NOT NULL,

	producer_long_change INTEGER NOT NULL,
	producer_short_change INTEGER NOT NULL,

	swapderlaers_long_change INTEGER NOT NULL,
	swapderlaers_short_change INTEGER NOT NULL,
	swapderlaers_spreading_change INTEGER NOT NULL,

	managedmoney_long_change INTEGER NOT NULL,
	managedmoney_short_change INTEGER NOT NULL,
	managedmoney_spreading_change INTEGER NOT NULL,

	other_long_change INTEGER NOT NULL,
	other_short_change INTEGER NOT NULL,
	other_spreading_change INTEGER NOT NULL,

	report_date DATE NOT NULL,
	id SERIAL PRIMARY KEY
);