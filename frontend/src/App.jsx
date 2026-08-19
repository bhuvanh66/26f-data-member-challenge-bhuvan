import { useEffect, useMemo, useState } from 'react'
import {
  Box, Container, Heading, Text, VStack, HStack, Slider, NativeSelect, RadioCard,
  Stat, Spinner, Separator, Tabs, Badge, Tooltip,
} from '@chakra-ui/react'

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

function OptionSlider({ label, options, index, onChange }) {
  return (
    <Slider.Root
      width="100%"
      min={0}
      max={options.length - 1}
      step={1}
      value={[index]}
      onValueChange={(d) => onChange(d.value[0])}
    >
      <HStack justify="space-between">
        <Slider.Label>{label}</Slider.Label>
        <Text fontSize="sm" color="fg.muted">{options[index]}</Text>
      </HStack>
      <Slider.Control>
        <Slider.Track>
          <Slider.Range />
        </Slider.Track>
        <Slider.Thumb index={0}>
          <Slider.HiddenInput />
        </Slider.Thumb>
      </Slider.Control>
    </Slider.Root>
  )
}

function NumberSlider({ label, max, value, onChange, suffix = ' yrs' }) {
  return (
    <Slider.Root
      width="100%"
      min={0}
      max={max}
      step={1}
      value={[value]}
      onValueChange={(d) => onChange(d.value[0])}
    >
      <HStack justify="space-between">
        <Slider.Label>{label}</Slider.Label>
        <Text fontSize="sm" color="fg.muted">{value}{suffix}</Text>
      </HStack>
      <Slider.Control>
        <Slider.Track>
          <Slider.Range />
        </Slider.Track>
        <Slider.Thumb index={0}>
          <Slider.HiddenInput />
        </Slider.Thumb>
      </Slider.Control>
    </Slider.Root>
  )
}

function OptionSelect({ label, options, value, onChange }) {
  return (
    <Box>
      <Text mb={1} fontWeight="medium">{label}</Text>
      <NativeSelect.Root>
        <NativeSelect.Field value={value} onChange={(e) => onChange(e.target.value)}>
          {options.map((o) => <option key={o} value={o}>{o}</option>)}
        </NativeSelect.Field>
        <NativeSelect.Indicator />
      </NativeSelect.Root>
    </Box>
  )
}

function OptionCards({ label, options, value, onChange }) {
  return (
    <RadioCard.Root value={value} onValueChange={(d) => onChange(d.value)} orientation="horizontal">
      <RadioCard.Label>{label}</RadioCard.Label>
      <HStack wrap="wrap" gap={2}>
        {options.map((o) => (
          <RadioCard.Item key={o} value={o} flex="1" minW="140px">
            <RadioCard.ItemHiddenInput />
            <RadioCard.ItemControl>
              <RadioCard.ItemText fontSize="sm">{o}</RadioCard.ItemText>
            </RadioCard.ItemControl>
          </RadioCard.Item>
        ))}
      </HStack>
    </RadioCard.Root>
  )
}

function fmtUsd(n) {
  return n == null ? '—' : `$${n.toLocaleString()}`
}

// Compact form for axis ticks and in-bar labels, where "$149,999" would collide
// with its neighbors — "$150k" carries the same information in a third the width.
function fmtCompact(n) {
  if (n >= 1000) return `$${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k`
  return `$${Math.round(n)}`
}

function CountryHistogram({ block, predictedUsd }) {
  const { histogram } = block
  const { edges, counts } = histogram
  const max = Math.max(...counts, 1)
  const [hovered, setHovered] = useState(null)

  return (
    <Box>
      <HStack align="end" gap={1} height="110px">
        {counts.map((c, i) => {
          const lo = edges[i]
          const hi = edges[i + 1]
          const isPrediction = predictedUsd >= lo && predictedUsd < hi
          const isHovered = hovered === i
          return (
            <Tooltip.Root key={i} openDelay={80} closeDelay={40}>
              <Tooltip.Trigger asChild>
                <VStack
                  gap={1}
                  flex="1"
                  height="100%"
                  justify="end"
                  cursor="default"
                  tabIndex={0}
                  data-histogram-bar={i}
                  onMouseEnter={() => setHovered(i)}
                  onMouseLeave={() => setHovered(null)}
                  onFocus={() => setHovered(i)}
                  onBlur={() => setHovered(null)}
                >
                  <Text fontSize="10px" color="fg.muted" lineHeight="1">{c}</Text>
                  <Box
                    width="100%"
                    height={`${Math.max((c / max) * 100, 3)}%`}
                    bg={isPrediction ? 'blue.500' : isHovered ? 'bg.emphasized' : 'bg.muted'}
                    outline={isHovered ? '2px solid' : 'none'}
                    outlineColor="blue.400"
                    borderRadius="sm"
                    transition="background 0.1s"
                  />
                </VStack>
              </Tooltip.Trigger>
              <Tooltip.Positioner>
                <Tooltip.Content>
                  <Text fontWeight="semibold">{c} respondent{c === 1 ? '' : 's'}</Text>
                  <Text fontSize="xs">{fmtCompact(lo)} – {fmtCompact(hi)}</Text>
                  {isPrediction && <Text fontSize="xs" color="blue.300">your prediction is here</Text>}
                </Tooltip.Content>
              </Tooltip.Positioner>
            </Tooltip.Root>
          )
        })}
      </HStack>
      <HStack justify="space-between" mt={1}>
        <Text fontSize="10px" color="fg.muted">{fmtCompact(edges[0])}</Text>
        <Text fontSize="10px" color="fg.muted">{fmtCompact(edges[edges.length - 1])}</Text>
      </HStack>
    </Box>
  )
}

export default function App() {
  const [meta, setMeta] = useState(null)
  const [metaError, setMetaError] = useState(null)

  const [yearsCode, setYearsCode] = useState(8)
  const [workExp, setWorkExp] = useState(6)
  const [ageIdx, setAgeIdx] = useState(1)
  const [edLevel, setEdLevel] = useState(null)
  const [orgSize, setOrgSize] = useState(null)
  const [remoteWork, setRemoteWork] = useState(null)
  const [icOrPm, setIcOrPm] = useState(null)
  const [country, setCountry] = useState('United States')
  const [devType, setDevType] = useState('Developer, full-stack')

  const [result, setResult] = useState(null)
  const [context, setContext] = useState(null)
  const [loading, setLoading] = useState(false)
  const [predictError, setPredictError] = useState(null)

  useEffect(() => {
    fetch(`${API}/meta`)
      .then((r) => r.json())
      .then((m) => {
        setMeta(m)
        setEdLevel(m.ed_level_options[4])
        setOrgSize(m.org_size_options[3])
        setRemoteWork(m.remote_options[3])
        setIcOrPm(m.ic_or_pm_options[0])
      })
      .catch((e) => setMetaError(String(e)))
  }, [])

  const answers = useMemo(() => {
    if (!meta || !edLevel) return null
    return {
      years_code: yearsCode,
      work_exp: workExp,
      age: meta.age_options[ageIdx],
      ed_level: edLevel,
      org_size: orgSize,
      remote_work: remoteWork,
      ic_or_pm: icOrPm,
      country,
      dev_type: devType,
    }
  }, [meta, yearsCode, workExp, ageIdx, edLevel, orgSize, remoteWork, icOrPm, country, devType])

  useEffect(() => {
    if (!answers) return
    setLoading(true)
    const timer = setTimeout(() => {
      fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(answers),
      })
        .then((r) => r.json())
        .then((d) => { setResult(d); setPredictError(null) })
        .catch((e) => setPredictError(String(e)))
        .finally(() => setLoading(false))
    }, 250)
    return () => clearTimeout(timer)
  }, [answers])

  useEffect(() => {
    if (!result) return
    const timer = setTimeout(() => {
      const params = new URLSearchParams({ country, predicted_usd: result.predicted_usd })
      fetch(`${API}/context?${params}`)
        .then((r) => r.json())
        .then(setContext)
        .catch(() => setContext(null))
    }, 250)
    return () => clearTimeout(timer)
  }, [country, result?.predicted_usd])

  if (metaError) {
    return (
      <Container maxW="lg" py={16}>
        <Text color="fg.error">
          Could not reach the API at {API}. Is `uvicorn ml.api:app` running? ({metaError})
        </Text>
      </Container>
    )
  }
  if (!meta || !edLevel) {
    return (
      <Container maxW="lg" py={16}>
        <Spinner />
      </Container>
    )
  }

  return (
    <Container maxW="lg" py={10}>
      <VStack align="stretch" gap={6}>
        <Box>
          <Heading size="lg">Salary predictor</Heading>
          <Text color="fg.muted">Enter your details. The prediction updates live.</Text>
        </Box>

        <Tabs.Root defaultValue="predict" variant="line">
          <Tabs.List>
            <Tabs.Trigger value="predict">Predict</Tabs.Trigger>
            <Tabs.Trigger value="data">Data</Tabs.Trigger>
          </Tabs.List>

          <Tabs.Content value="predict">
            <VStack align="stretch" gap={8} pt={4}>
              <VStack align="stretch" gap={6}>
                <OptionSelect label="Country" options={meta.country_options} value={country} onChange={setCountry} />
                <OptionSelect label="Role" options={meta.dev_type_options} value={devType} onChange={setDevType} />
                <NumberSlider label="Years coding" max={meta.years_code_max} value={yearsCode} onChange={setYearsCode} />
                <NumberSlider label="Years of professional work experience" max={meta.work_exp_max} value={workExp} onChange={setWorkExp} />
                <OptionSlider label="Age" options={meta.age_options} index={ageIdx} onChange={setAgeIdx} />
                <OptionSelect label="Education" options={meta.ed_level_options} value={edLevel} onChange={setEdLevel} />
                <OptionSelect label="Organization size" options={meta.org_size_options} value={orgSize} onChange={setOrgSize} />
                <OptionCards label="Remote work" options={meta.remote_options} value={remoteWork} onChange={setRemoteWork} />
                <OptionCards label="Role type" options={meta.ic_or_pm_options} value={icOrPm} onChange={setIcOrPm} />
              </VStack>

              <Separator />

              <Box>
                {predictError && <Text color="fg.error">{predictError}</Text>}
                {result && (
                  <VStack align="stretch" gap={1}>
                    <Stat.Root>
                      <Stat.Label>Predicted annual salary {loading && <Spinner size="xs" ml={2} />}</Stat.Label>
                      <Stat.ValueText fontSize="4xl">{fmtUsd(result.predicted_usd)}</Stat.ValueText>
                    </Stat.Root>
                    <Text color="fg.muted" fontSize="sm">
                      50% interval: {fmtUsd(result.interval_50_usd[0])} – {fmtUsd(result.interval_50_usd[1])}
                    </Text>
                    <Text color="fg.muted" fontSize="sm">
                      80% interval: {fmtUsd(result.interval_80_usd[0])} – {fmtUsd(result.interval_80_usd[1])}
                    </Text>
                    <Text color="fg.muted" fontSize="xs" mt={2}>model: {result.model_name}</Text>
                  </VStack>
                )}
              </Box>
            </VStack>
          </Tabs.Content>

          <Tabs.Content value="data">
            <VStack align="stretch" gap={6} pt={4}>
              {!context && <Text color="fg.muted">Waiting for a prediction…</Text>}

              {context?.country && (
                <Box borderWidth="1px" borderRadius="md" p={4}>
                  <HStack justify="space-between" mb={2}>
                    <Heading size="sm">{context.country.name}</Heading>
                    <Badge colorPalette="blue">{context.country.n} respondents</Badge>
                  </HStack>
                  <Text fontSize="sm" color="fg.muted" mb={3}>
                    Your prediction ({fmtUsd(result?.predicted_usd)}) is higher than{' '}
                    <b>{context.country.percentile.toFixed(0)}%</b> of respondents from{' '}
                    {context.country.name} (country median: {fmtUsd(context.country.median_usd)}).
                  </Text>
                  <CountryHistogram block={context.country} predictedUsd={result?.predicted_usd} />
                  <Text fontSize="xs" color="fg.muted" mt={1}>
                    Distribution of actual reported salaries in {context.country.name}; the blue bar is
                    where your prediction falls. Hover or focus any bar for its exact range.
                  </Text>
                </Box>
              )}
              {context && !context.country && (
                <Text color="fg.muted" fontSize="sm">
                  Not enough respondents from this country to show a distribution.
                </Text>
              )}
            </VStack>
          </Tabs.Content>
        </Tabs.Root>
      </VStack>
    </Container>
  )
}
