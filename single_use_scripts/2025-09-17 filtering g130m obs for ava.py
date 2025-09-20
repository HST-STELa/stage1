from stage1_processing import preloads
import paths

hosts = preloads.hosts.copy()

has_g130m_lya = hosts['n_cos_g130m_lya_obs'].filled(False) > 0
has_g140m_lya = hosts['n_stis_g140m_lya_obs'].filled(False) > 0
has_e140m_lya = hosts['n_stis_e140m_lya_obs'].filled(False) > 0
unverified = hosts['lya_verified'].mask

keepers = has_g130m_lya & ~(has_g140m_lya | has_e140m_lya) & unverified

hosts = hosts[keepers]

hosts.write(paths.data / 'packages/outbox' / '2025-09-17_hosts_with_only_g130m_lya_data.ecsv')

# make target list to use with download_data script to pull data
names = preloads.stela_names
targets = names.loc['hostname', hosts['hostname']]['hostname_file']
print(targets.tolist())