/** Search form validation. Limits match SearchRequest in app/api/schemas.py. */
import { z } from 'zod'

export const searchFormSchema = z.object({
  query: z
    .string()
    .trim()
    .min(3, 'Please enter at least 3 characters.')
    .max(2000, 'Query must be 2000 characters or fewer.'),
  collection: z.string().min(1, 'Select a collection to search.'),
  useCache: z.boolean(),
})

export type SearchFormValues = z.infer<typeof searchFormSchema>
